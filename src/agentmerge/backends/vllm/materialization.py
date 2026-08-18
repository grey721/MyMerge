"""In-memory LoRA fusion for the optional vLLM backend."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ...core.config import AdapterConfig
from ...core.policies import (
    FusionPolicy,
    ModuleLocator,
    TransformerModuleLocator,
)
from ..algebra import (
    FactorComposition,
    FusionAlgebra,
    FusionModuleSpec,
    SourceAdapterDescriptor,
    create_fusion_algebra,
)
from ..pool import ResidentAdapterPool
from ..resources import SourceAdapterResource


@dataclass(frozen=True)
class ResidentSourceAdapter:
    """One source LoRA held in host memory for the worker lifetime."""

    spec: AdapterConfig
    peft_config: Mapping[str, Any]
    tensors: Mapping[str, Any]

    @property
    def rank(self) -> int:
        return int(self.peft_config["r"])

    @property
    def native_scale(self) -> float:
        alpha = float(self.peft_config["lora_alpha"])
        if bool(self.peft_config.get("use_rslora", False)):
            return alpha / math.sqrt(self.rank)
        return alpha / self.rank


@dataclass
class MaterializedTensorLoRA:
    """Candidate tensors plus the PEFT metadata consumed by vLLM."""

    name: str
    rank: int
    peft_config: Mapping[str, Any]
    tensors: Dict[str, Any]
    source_keys: Tuple[str, ...] = ()

    @property
    def resident_bytes(self) -> int:
        return sum(_tensor_nbytes(item) for item in self.tensors.values())

    @property
    def device(self) -> Optional[str]:
        if not self.tensors:
            return None
        return str(next(iter(self.tensors.values())).device)

    def release(self) -> None:
        self.tensors.clear()


@dataclass(frozen=True)
class _VllmModuleExecution:
    """Pre-resolved tensor keys and algebra target for one LoRA module."""

    a_key: str
    b_key: str
    spec: FusionModuleSpec


class _VllmResidentAdapter(SourceAdapterResource):
    """One independently releasable tensor LoRA and vLLM registration."""

    def __init__(
        self, coordinate_index: int, source: ResidentSourceAdapter
    ) -> None:
        self._coordinate_index = int(coordinate_index)
        self._descriptor = SourceAdapterDescriptor(
            source.spec.name, source.rank, source.native_scale
        )
        self._source = source
        self._transport = None
        self._request = None
        self._closed = False
        self._registered = False
        self.registration_count = 0
        self.registration_seconds = 0.0
        self.unregistration_count = 0
        self.unregistration_seconds = 0.0

    @property
    def adapter_key(self) -> str:
        return self._source.spec.name

    @property
    def coordinate_index(self) -> int:
        return self._coordinate_index

    @property
    def spec(self) -> AdapterConfig:
        return self._source.spec

    @property
    def descriptor(self) -> SourceAdapterDescriptor:
        return self._descriptor

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def source(self) -> ResidentSourceAdapter:
        if self._closed:
            raise RuntimeError(
                "vLLM source Adapter {!r} has been released".format(
                    self.adapter_key
                )
            )
        return self._source

    @property
    def resident_bytes(self) -> int:
        if self._closed:
            return 0
        return sum(_tensor_nbytes(item) for item in self._source.tensors.values())

    @property
    def device(self):
        if self._closed or not self._source.tensors:
            return None
        return str(next(iter(self._source.tensors.values())).device)

    @property
    def registered(self) -> bool:
        return self._registered

    def move_to(self, device: str) -> None:
        source = self.source
        if not isinstance(source.tensors, dict):
            raise TypeError("Resident source tensor mapping must be mutable")
        for key, tensor in tuple(source.tensors.items()):
            source.tensors[key] = tensor.to(device=device)

    def bind_transport(self, transport: Any) -> None:
        if self._closed:
            raise RuntimeError("Cannot bind a released vLLM source Adapter")
        if self._transport is not None:
            if self._transport is transport:
                return
            raise RuntimeError("vLLM source Adapter is already bound")
        self._transport = transport
        self._request = transport.make_request(
            name="source_{}".format(self.adapter_key),
            request_id=self.coordinate_index + 1,
            peft_config=self._source.peft_config,
            tensors=self._source.tensors,
        )

    def request(self) -> Any:
        if self._closed:
            raise RuntimeError(
                "vLLM source Adapter {!r} has been released".format(
                    self.adapter_key
                )
            )
        if self._transport is None or self._request is None:
            raise RuntimeError("vLLM source Adapter is not bound to a transport")
        if self._registered:
            return self._request
        started = time.perf_counter()
        try:
            registered = self._transport.register(self._request)
        finally:
            self.registration_seconds += time.perf_counter() - started
            self.registration_count += 1
        if registered is False:
            raise RuntimeError(
                "vLLM rejected source LoRA {!r}".format(self.adapter_key)
            )
        self._registered = True
        return self._request

    def close(self) -> None:
        if self._closed:
            return
        if self._registered:
            started = time.perf_counter()
            try:
                removed = self._transport.unregister(
                    int(self._request.lora_int_id)
                )
            finally:
                self.unregistration_seconds += time.perf_counter() - started
                self.unregistration_count += 1
            if removed is False:
                raise RuntimeError(
                    "vLLM source Adapter {!r} was not registered".format(
                        self.adapter_key
                    )
                )
            self._registered = False
        if self._request is not None:
            tensors = getattr(self._request, "lora_tensors", None)
            if isinstance(tensors, dict):
                tensors.clear()
        if isinstance(self._source.tensors, dict):
            self._source.tensors.clear()
        self._request = None
        self._transport = None
        self._closed = True


class VllmResidentAdapterPool(ResidentAdapterPool):
    """Own CUDA/CPU source tensors and their persistent vLLM registrations."""

    resource_kind = "vllm_tensor"

    def __init__(
        self,
        adapters: Sequence[AdapterConfig],
        *,
        auto_seal: bool = True,
    ) -> None:
        self._transport = None
        super().__init__((), auto_seal=False)
        for adapter in adapters:
            self.add_source(adapter)
        if auto_seal:
            self.seal()

    @property
    def sources(self) -> Tuple[ResidentSourceAdapter, ...]:
        self._require_search_ready()
        return tuple(
            self._resource(key).source for key in self.resident_adapter_keys
        )

    def move_to(self, device: str) -> None:
        """Move every resident source factor once after engine allocation."""

        self._require_search_ready()
        normalized = str(device).lower()
        if normalized not in ("cpu", "cuda", "cuda:0"):
            raise ValueError(
                "vllm.source_adapter_device must be cpu or cuda"
            )
        for key in self.resident_adapter_keys:
            self._resource(key).move_to(normalized)
        if normalized.startswith("cuda"):
            import torch

            torch.cuda.synchronize()

    def source(self, name: str) -> ResidentSourceAdapter:
        self._require_search_ready()
        resource = self._resource(name)
        if resource.closed:
            raise KeyError("Adapter {!r} is not resident".format(name))
        return resource.source

    def bind_transport(self, transport: Any) -> None:
        """Bind source registration to one engine-neutral transport."""

        self._require_search_ready()
        if self._transport is not None:
            if self._transport is transport:
                return
            raise RuntimeError("vLLM Adapter pool is already bound to a transport")
        self._transport = transport
        for key in self.resident_adapter_keys:
            self._resource(key).bind_transport(transport)

    def request_for_source(self, name: str) -> Any:
        """Return a persistent source request, registering it on first use."""

        self._require_search_ready()
        return self._resource(name).request()

    def runtime_snapshot(self) -> Mapping[str, Any]:
        snapshot = dict(super().runtime_snapshot())
        snapshot.update(
            {
                "transport_bound": self._transport is not None,
                "registered_source_adapter_names": [
                    resource.adapter_key
                    for resource in self._all_resources()
                    if resource.registered
                ],
                "registration_count": sum(
                    item.registration_count for item in self._all_resources()
                ),
                "registration_seconds": sum(
                    item.registration_seconds for item in self._all_resources()
                ),
                "unregistration_count": sum(
                    item.unregistration_count for item in self._all_resources()
                ),
                "unregistration_seconds": sum(
                    item.unregistration_seconds for item in self._all_resources()
                ),
            }
        )
        return snapshot

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self._transport = None

    def _create_source_resource(
        self, coordinate_index: int, adapter: AdapterConfig
    ) -> SourceAdapterResource:
        source = _load_source(adapter)
        existing = tuple(
            self._resource(key).source for key in self.resident_adapter_keys
        )
        _validate_pool(existing + (source,))
        return _VllmResidentAdapter(coordinate_index, source)

    def _resource(self, adapter_key: str) -> _VllmResidentAdapter:
        return self._resources[adapter_key]

    def _all_resources(self) -> Tuple[_VllmResidentAdapter, ...]:
        return tuple(self._resource(key) for key in self.adapter_keys)


def _load_source(adapter: AdapterConfig) -> ResidentSourceAdapter:
    directory = Path(adapter.path)
    config_path = directory / "adapter_config.json"
    tensor_path = directory / "adapter_model.safetensors"
    if not config_path.is_file():
        raise ValueError(
            "Adapter {!r} has no adapter_config.json".format(adapter.name)
        )
    if not tensor_path.is_file():
        raise ValueError(
            "vLLM memory transport requires adapter_model.safetensors for "
            "adapter {!r}".format(adapter.name)
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - vLLM installs it
        from ..base import OptionalDependencyError

        raise OptionalDependencyError(
            "The vLLM backend requires safetensors. Install the project "
            "with: pip install -e '.[vllm]'"
        ) from exc
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensor_path), device="cpu")
    return ResidentSourceAdapter(adapter, config, tensors)


def _validate_pool(sources: Sequence[ResidentSourceAdapter]) -> None:
    if not sources:
        raise ValueError("vLLM source pool cannot be empty")
    first_keys = frozenset(sources[0].tensors)
    first_targets = _target_modules(sources[0].peft_config)
    for source in sources:
        config = source.peft_config
        if str(config.get("peft_type", "")).upper() != "LORA":
            raise ValueError(
                "Adapter {!r} is not a LoRA".format(source.spec.name)
            )
        if source.rank <= 0:
            raise ValueError(
                "Adapter {!r} has invalid rank {}".format(
                    source.spec.name, source.rank
                )
            )
        if config.get("rank_pattern") or config.get("alpha_pattern"):
            raise ValueError("vLLM tensor fusion requires uniform rank and alpha")
        if bool(config.get("use_dora", False)):
            raise ValueError("vLLM tensor fusion does not support DoRA")
        if config.get("modules_to_save"):
            raise ValueError(
                "vLLM tensor fusion does not support modules_to_save"
            )
        if str(config.get("bias", "none")).lower() != "none" or bool(
            config.get("lora_bias", False)
        ):
            raise ValueError("vLLM tensor fusion requires bias='none'")
        if frozenset(source.tensors) != first_keys:
            raise ValueError(
                "All source adapters must cover identical LoRA tensor keys"
            )
        if _target_modules(config) != first_targets:
            raise ValueError(
                "All source adapters must use identical target_modules"
            )
    if not first_keys or any(
        not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight"))
        for key in first_keys
    ):
        raise ValueError(
            "vLLM tensor fusion supports factor-only PEFT safetensors"
        )


class VllmTensorAdapterMaterializer:
    """Execute a fusion algebra plan as an in-memory vLLM LoRA.

    Module paths are resolved once when the resident worker starts. Candidate
    compilation then preallocates final tensors and executes concatenated or
    shared rank slices in place, avoiding repeated path parsing, ``torch.cat``
    and scaled-factor intermediate tensors on the hot path.
    """

    def __init__(
        self,
        pool: VllmResidentAdapterPool,
        algebra: Optional[FusionAlgebra] = None,
        module_locator: Optional[ModuleLocator] = None,
    ) -> None:
        self.pool = pool
        self.algebra = algebra or create_fusion_algebra()
        self._module_locator = module_locator or TransformerModuleLocator()
        self._execution_modules = self._precompile_execution_modules()
        self._module_specs = tuple(
            item.spec for item in self._execution_modules
        )
        self._execution_by_key = {
            item.spec.key: item for item in self._execution_modules
        }
        self._materialization_count = 0
        self._materialization_seconds = 0.0
        self._plan_compilation_seconds = 0.0
        self._release_count = 0
        self._release_seconds = 0.0
        self._target_allocation_count = 0
        self._copied_module_count = 0
        self._factor_copy_operations = 0
        self._factor_bytes_copied = 0
        self._last_materialization: Dict[str, Any] = {}

    def materialize(
        self, policy: FusionPolicy, adapter_name: str
    ) -> MaterializedTensorLoRA:
        started = time.perf_counter()
        compile_started = time.perf_counter()
        plan = self.algebra.compile(
            policy,
            self.pool.descriptors,
            self._module_specs,
        )
        compilation_seconds = time.perf_counter() - compile_started
        self._plan_compilation_seconds += compilation_seconds
        total_rank = plan.total_rank
        tensors: Dict[str, Any] = {}
        active_sources = tuple(
            self.pool.source(item.source_name) for item in plan.slices
        )
        first = active_sources[0]
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - vLLM installs torch
            from ..base import OptionalDependencyError

            raise OptionalDependencyError(
                "The vLLM backend requires torch. Install: pip install -e '.[vllm]'"
            ) from exc

        bytes_copied = 0
        allocation_count = 0
        copy_operations = 0
        zero_coefficient_slices = 0
        with torch.inference_mode():
            for instruction in plan.modules:
                execution = self._execution_by_key[instruction.module_key]
                first_a = first.tensors[execution.a_key]
                first_b = first.tensors[execution.b_key]
                target_a = torch.empty(
                    (total_rank, int(first_a.shape[1])),
                    dtype=first_a.dtype,
                    device=first_a.device,
                )
                target_b = torch.empty(
                    (int(first_b.shape[0]), total_rank),
                    dtype=first_b.dtype,
                    device=first_b.device,
                )
                allocation_count += 2
                if plan.factor_composition is FactorComposition.LINEAR_SUM:
                    target_a.zero_()
                    target_b.zero_()
                for factor_slice, a_coefficient, b_coefficient, source in zip(
                    plan.slices,
                    instruction.a_coefficients,
                    instruction.b_coefficients,
                    active_sources,
                ):
                    source_a = source.tensors[execution.a_key]
                    source_b = source.tensors[execution.b_key]
                    if int(source_a.shape[0]) != factor_slice.rank:
                        raise ValueError(
                            "Adapter {!r} A rank does not match config".format(
                                source.spec.name
                            )
                        )
                    if int(source_b.shape[1]) != factor_slice.rank:
                        raise ValueError(
                            "Adapter {!r} B rank does not match config".format(
                                source.spec.name
                            )
                        )
                    if (
                        int(source_a.shape[1]) != int(target_a.shape[1])
                        or int(source_b.shape[0]) != int(target_b.shape[0])
                    ):
                        raise ValueError(
                            "Adapter {!r} has incompatible module shapes".format(
                                source.spec.name
                            )
                        )
                    start, end = factor_slice.start, factor_slice.end
                    target_a_slice = target_a[start:end]
                    target_b_slice = target_b[:, start:end]
                    if plan.factor_composition is FactorComposition.CONCATENATE:
                        _copy_scaled_tensor(
                            torch, target_a_slice, source_a, a_coefficient
                        )
                        _copy_scaled_tensor(
                            torch, target_b_slice, source_b, b_coefficient
                        )
                    else:
                        _add_scaled_tensor(target_a_slice, source_a, a_coefficient)
                        _add_scaled_tensor(target_b_slice, source_b, b_coefficient)
                    if a_coefficient != 0.0:
                        bytes_copied += _tensor_nbytes(source_a)
                    copy_operations += 1
                    if b_coefficient != 0.0:
                        bytes_copied += _tensor_nbytes(source_b)
                    if a_coefficient == 0.0 or b_coefficient == 0.0:
                        zero_coefficient_slices += 1
                    copy_operations += 1
                tensors[execution.a_key] = target_a
                tensors[execution.b_key] = target_b

        peft_config = dict(first.peft_config)
        peft_config.update(
            {
                "r": total_rank,
                "lora_alpha": total_rank,
                "lora_dropout": 0.0,
                "use_rslora": False,
                "rank_pattern": {},
                "alpha_pattern": {},
                "init_lora_weights": False,
            }
        )
        elapsed = time.perf_counter() - started
        self._materialization_count += 1
        self._materialization_seconds += elapsed
        self._target_allocation_count += allocation_count
        self._copied_module_count += len(plan.modules)
        self._factor_copy_operations += copy_operations
        self._factor_bytes_copied += bytes_copied
        self._last_materialization = {
            "adapter_name": adapter_name,
            "algebra": plan.algebra,
            "factor_composition": plan.factor_composition.value,
            "active_adapter_count": len(plan.slices),
            "total_rank": total_rank,
            "tensor_count": len(tensors),
            "precompiled_module_count": len(self._execution_modules),
            "plan_compilation_seconds": compilation_seconds,
            "target_allocation_count": allocation_count,
            "copied_module_count": len(plan.modules),
            "factor_copy_operations": copy_operations,
            "factor_bytes_copied": bytes_copied,
            "zero_coefficient_slices": zero_coefficient_slices,
            "total_seconds": elapsed,
        }
        return MaterializedTensorLoRA(
            name=adapter_name,
            rank=total_rank,
            peft_config=peft_config,
            tensors=tensors,
            source_keys=plan.active_adapter_names,
        )

    def release(self, adapter: MaterializedTensorLoRA) -> None:
        started = time.perf_counter()
        adapter.release()
        elapsed = time.perf_counter() - started
        self._release_count += 1
        self._release_seconds += elapsed

    def runtime_snapshot(self) -> Mapping[str, Any]:
        return {
            "algebra": self.algebra.name,
            "materialization_count": self._materialization_count,
            "materialization_seconds": self._materialization_seconds,
            "plan_compilation_seconds": self._plan_compilation_seconds,
            "precompiled_module_count": len(self._execution_modules),
            "target_allocation_count": self._target_allocation_count,
            "copied_module_count": self._copied_module_count,
            "factor_copy_operations": self._factor_copy_operations,
            "factor_bytes_copied": self._factor_bytes_copied,
            "last_materialization": dict(self._last_materialization),
            "release_count": self._release_count,
            "release_seconds": self._release_seconds,
        }

    def _precompile_execution_modules(
        self,
    ) -> Tuple[_VllmModuleExecution, ...]:
        first = self.pool.sources[0]
        compiled = []
        for a_key in sorted(
            key for key in first.tensors if key.endswith(".lora_A.weight")
        ):
            b_key = a_key[: -len("lora_A.weight")] + "lora_B.weight"
            if b_key not in first.tensors:
                raise ValueError("Missing B factor for {!r}".format(a_key))
            module_path = a_key[: -len(".lora_A.weight")]
            compiled.append(
                _VllmModuleExecution(
                    a_key=a_key,
                    b_key=b_key,
                    spec=FusionModuleSpec(
                        key=module_path,
                        target=self._module_locator.locate(module_path),
                    ),
                )
            )
        if not compiled:
            raise ValueError("vLLM source pool has no LoRA factor modules")
        return tuple(compiled)


def supported_rank_ceiling(rank: int) -> int:
    """Return the smallest rank accepted by vLLM 0.10."""

    for supported in (8, 16, 32, 64, 128, 256, 320, 512):
        if rank <= supported:
            return supported
    raise ValueError("Fused LoRA rank {} exceeds vLLM maximum 512".format(rank))


def _target_modules(config: Mapping[str, Any]) -> frozenset:
    value = config.get("target_modules")
    if isinstance(value, str):
        return frozenset((value,))
    return frozenset(str(item) for item in (value or ()))


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _copy_scaled_tensor(
    torch_module: Any,
    target: Any,
    source: Any,
    coefficient: float,
) -> None:
    """Copy one source tensor into a disjoint target interval."""

    if coefficient == 0.0:
        target.zero_()
    elif coefficient == 1.0:
        target.copy_(source)
    else:
        torch_module.mul(source, float(coefficient), out=target)


def _add_scaled_tensor(target: Any, source: Any, coefficient: float) -> None:
    """Accumulate one source tensor into a shared target interval."""

    if coefficient != 0.0:
        target.add_(source, alpha=float(coefficient))


__all__ = [
    "MaterializedTensorLoRA",
    "ResidentSourceAdapter",
    "VllmResidentAdapterPool",
    "VllmTensorAdapterMaterializer",
    "supported_rank_ceiling",
]
