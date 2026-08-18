"""Candidate-scoped PEFT execution of backend-neutral fusion plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from ..core.config import AdapterConfig
from ..core.policies import (
    FusionPolicy,
    ModuleLocator,
    TransformerModuleLocator,
    active_adapter_names,
    uniform_policy_mixture,
)
from .algebra import (
    FactorComposition,
    FusionAlgebra,
    FusionExecutionPlan,
    FusionModuleSpec,
    SourceAdapterDescriptor,
    create_fusion_algebra,
)
from .peft_pool import PeftResidentAdapterPool
from .pool import AdapterPool, ResidentAdapterPool
from .resources import FusedAdapterResource, FusionAdapterFactory


BASE_ADAPTER_NAME = "__base_model__"


@dataclass(frozen=True)
class _PeftModuleExecution:
    key: str
    module: Any
    spec: FusionModuleSpec


class AdapterMaterializer(ABC):
    """Backend-specific compiler for one ephemeral fused adapter.

    The evolutionary layer only creates immutable fusion policies. Fusion
    mathematics belongs to :class:`FusionAlgebra`; this class is the PEFT
    tensor executor and lifecycle boundary only.
    """

    @abstractmethod
    def materialize(self, model, mixture: FusionPolicy, adapter_name: str) -> str:
        """Compile ``mixture`` into ``adapter_name`` and return its name."""

    @abstractmethod
    def release(self, model, adapter_name: str) -> None:
        """Release one compiled adapter from ``model``."""

    def runtime_snapshot(self) -> Mapping[str, Any]:
        """Return optional materialization profiling counters."""

        return {}


class PeftDirectAdapterMaterializer(AdapterMaterializer):
    """Execute a fusion-algebra plan into a temporary PEFT adapter.

    This deliberately does not call ``add_weighted_adapter``.  PEFT's helper
    first builds and registers another weighted adapter; here the configured
    algebra chooses the target rank, factor intervals and A/B coefficients.
    The backend allocates that adapter only when first used and executes the
    plan directly.  The temporary adapter always uses scale one because source
    scaling is already encoded in the plan.
    """

    def __init__(
        self,
        pool: Optional[PeftResidentAdapterPool] = None,
        algebra: Optional[FusionAlgebra] = None,
        module_locator: Optional[ModuleLocator] = None,
    ) -> None:
        self.pool = pool
        self.algebra = algebra or create_fusion_algebra()
        self._module_locator = module_locator or TransformerModuleLocator()
        self._precompiled_model_id = None  # type: Optional[int]
        self._precompiled_modules = ()  # type: Tuple[_PeftModuleExecution, ...]
        self._precompiled_specs = ()  # type: Tuple[FusionModuleSpec, ...]
        self._precompiled_by_key = {}  # type: Dict[str, Any]
        self._plan_compilation_seconds = 0.0
        self._materialization_count = 0
        self._materialization_seconds = 0.0
        self._last_materialization = {}  # type: Dict[str, Any]
        self._release_count = 0
        self._release_seconds = 0.0
        self._last_release_seconds = 0.0
        self._target_allocation_count = 0
        self._copied_module_count = 0
        self._factor_copy_operations = 0
        self._factor_bytes_copied = 0

    def materialize(self, model, mixture: FusionPolicy, adapter_name: str) -> str:
        total_start = time.perf_counter()
        preparation_start = time.perf_counter()
        peft_configs = getattr(model, "peft_config", None)
        if not isinstance(peft_configs, dict):
            raise TypeError("PEFT model does not expose peft_config")
        if self.pool is not None:
            if self.pool.model is not model:
                raise RuntimeError(
                    "PEFT materializer received a model outside its Adapter pool"
                )
            self.pool.validate_policy(mixture)
        active_names = active_adapter_names(mixture)
        if not active_names:
            raise ValueError("Cannot materialize an empty Adapter policy")
        try:
            policy_configs = {
                name: peft_configs[name] for name in active_names
            }
        except KeyError as exc:
            raise KeyError(
                "Source adapter {!r} is not loaded; every source adapter must "
                "be preloaded by PeftCausalLMBackend".format(exc.args[0])
            ) from exc
        if adapter_name in peft_configs:
            raise RuntimeError(
                "Temporary adapter {!r} is already registered".format(adapter_name)
            )

        self._validate_compatible_configs(
            tuple(policy_configs.values()), tuple(policy_configs)
        )
        if self.pool is not None:
            sources = self.pool.descriptors
        else:
            sources = tuple(
                _peft_source_descriptor(name, config)
                for name, config in policy_configs.items()
            )
        module_executions = self._precompile_execution_modules(model)
        compile_start = time.perf_counter()
        execution_plan = self.algebra.compile(
            mixture,
            sources,
            self._precompiled_specs,
        )
        compilation_seconds = time.perf_counter() - compile_start
        self._plan_compilation_seconds += compilation_seconds
        active_names = execution_plan.active_adapter_names
        source_configs = [policy_configs[name] for name in active_names]
        total_rank = execution_plan.total_rank
        temporary_config = _temporary_lora_config(source_configs[0], total_rank)
        preparation_seconds = time.perf_counter() - preparation_start

        try:
            _synchronize_model_device(model)
            allocation_start = time.perf_counter()
            # Meta initialization avoids PEFT's module-by-module CPU->GPU copy.
            # Empty factors are filled directly on the source device below.
            model.add_adapter(
                adapter_name=adapter_name,
                peft_config=temporary_config,
                low_cpu_mem_usage=True,
                autocast_adapter_dtype=False,
            )
            _synchronize_model_device(model)
            registration_seconds = time.perf_counter() - allocation_start

            target_allocation_start = time.perf_counter()
            target_allocation_count = self._materialize_empty_adapter_weights(
                model, active_names, adapter_name
            )
            _synchronize_model_device(model)
            target_allocation_seconds = (
                time.perf_counter() - target_allocation_start
            )

            copy_start = time.perf_counter()
            copy_stats = self._execute_factor_plan(
                execution_plan,
                adapter_name,
            )
            _synchronize_model_device(model)
            copy_seconds = time.perf_counter() - copy_start
        except Exception as exc:
            if _adapter_is_registered(model, adapter_name):
                try:
                    self.release(model, adapter_name)
                except Exception as cleanup_exc:
                    raise RuntimeError(
                        "Temporary adapter {!r} failed to materialize and "
                        "cleanup also failed: {}".format(adapter_name, cleanup_exc)
                    ) from exc
            raise
        total_seconds = time.perf_counter() - total_start
        profile = {
            "adapter_name": adapter_name,
            "algebra": execution_plan.algebra,
            "factor_composition": execution_plan.factor_composition.value,
            "active_adapter_count": len(active_names),
            "total_rank": total_rank,
            "plan_compilation_seconds": compilation_seconds,
            "precompiled_module_count": len(module_executions),
            "preparation_seconds": preparation_seconds,
            "registration_seconds": registration_seconds,
            "target_allocation_seconds": target_allocation_seconds,
            "factor_copy_seconds": copy_seconds,
            "total_seconds": total_seconds,
            "target_allocation_count": target_allocation_count,
        }
        profile.update(copy_stats)
        self._materialization_count += 1
        self._materialization_seconds += total_seconds
        self._target_allocation_count += target_allocation_count
        self._copied_module_count += copy_stats["copied_module_count"]
        self._factor_copy_operations += copy_stats["factor_copy_operations"]
        self._factor_bytes_copied += copy_stats["factor_bytes_copied"]
        self._last_materialization = profile
        return adapter_name

    def release(self, model, adapter_name: str) -> None:
        _synchronize_model_device(model)
        start = time.perf_counter()
        if not hasattr(model, "delete_adapter"):
            raise TypeError("PEFT model does not support adapter deletion")
        model.delete_adapter(adapter_name)
        _synchronize_model_device(model)
        elapsed = time.perf_counter() - start
        self._release_count += 1
        self._release_seconds += elapsed
        self._last_release_seconds = elapsed

    def runtime_snapshot(self) -> Mapping[str, Any]:
        return {
            "algebra": self.algebra.name,
            "materialization_count": self._materialization_count,
            "materialization_seconds": self._materialization_seconds,
            "plan_compilation_seconds": self._plan_compilation_seconds,
            "precompiled_module_count": len(self._precompiled_modules),
            "target_allocation_count": self._target_allocation_count,
            "copied_module_count": self._copied_module_count,
            "factor_copy_operations": self._factor_copy_operations,
            "factor_bytes_copied": self._factor_bytes_copied,
            "last_materialization": dict(self._last_materialization),
            "release_count": self._release_count,
            "release_seconds": self._release_seconds,
            "last_release_seconds": self._last_release_seconds,
        }

    def _precompile_execution_modules(
        self, model
    ) -> Tuple[_PeftModuleExecution, ...]:
        """Resolve stable PEFT module paths once per resident model."""

        model_id = id(model)
        if self._precompiled_model_id == model_id:
            return self._precompiled_modules
        named_modules = getattr(model, "named_modules", None)
        modules = getattr(model, "modules", None)
        if not callable(named_modules) and not callable(modules):
            compiled = ()
        else:
            compiled_items = []
            for module_path, module in _iter_named_modules(model):
                lora_a = getattr(module, "lora_A", None)
                lora_b = getattr(module, "lora_B", None)
                if lora_a is None or lora_b is None:
                    continue
                key = str(module_path)
                compiled_items.append(
                    _PeftModuleExecution(
                        key=key,
                        module=module,
                        spec=FusionModuleSpec(
                            key=key,
                            target=self._module_locator.locate(key, module),
                        ),
                    )
                )
            compiled = tuple(compiled_items)
        self._precompiled_model_id = model_id
        self._precompiled_modules = compiled
        self._precompiled_specs = tuple(item.spec for item in compiled)
        self._precompiled_by_key = {
            item.key: item.module for item in compiled
        }
        return compiled

    @staticmethod
    def _materialize_empty_adapter_weights(
        model,
        active_names: Sequence[str],
        temporary_name: str,
    ) -> int:
        """Replace meta LoRA factors with empty tensors on the source device."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - real backend requires torch
            raise RuntimeError("Direct PEFT materialization requires torch") from exc

        allocated = 0
        for module in model.modules():
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if (
                lora_a is None
                or lora_b is None
                or temporary_name not in lora_a
                or temporary_name not in lora_b
            ):
                continue
            source_name = next(
                (
                    name
                    for name in active_names
                    if name in lora_a and name in lora_b
                ),
                None,
            )
            if source_name is None:
                continue
            source_a = _factor_parameter(lora_a[source_name])
            source_b = _factor_parameter(lora_b[source_name])
            target_a = _factor_parameter(lora_a[temporary_name])
            target_b = _factor_parameter(lora_b[temporary_name])
            if target_a.device.type != "meta" or target_b.device.type != "meta":
                continue
            target_a_module = lora_a[temporary_name]
            target_b_module = lora_b[temporary_name]
            target_a_module.weight = torch.nn.Parameter(
                torch.empty(
                    tuple(target_a.shape),
                    device=source_a.device,
                    dtype=source_a.dtype,
                ),
                requires_grad=False,
            )
            target_b_module.weight = torch.nn.Parameter(
                torch.empty(
                    tuple(target_b.shape),
                    device=source_b.device,
                    dtype=source_b.dtype,
                ),
                requires_grad=False,
            )
            allocated += 2
        return allocated

    @staticmethod
    def _config_rank(config: Any) -> int:
        rank_pattern = getattr(config, "rank_pattern", None) or {}
        if rank_pattern:
            raise ValueError(
                "Per-layer rank_pattern is not supported by the direct "
                "materializer; use adapters with a uniform rank per module"
            )
        alpha_pattern = getattr(config, "alpha_pattern", None) or {}
        if alpha_pattern:
            raise ValueError(
                "Per-layer alpha_pattern is not supported by the direct "
                "materializer; use adapters with uniform native scaling"
            )
        return int(getattr(config, "r", 0))

    @staticmethod
    def _validate_compatible_configs(
        configs: Sequence[Any], active_names: Sequence[str]
    ) -> None:
        first = configs[0]
        first_targets = _target_module_set(getattr(first, "target_modules", None))
        if not first_targets:
            raise ValueError("Direct materialization requires target_modules")
        for name, config in zip(active_names, configs):
            targets = _target_module_set(getattr(config, "target_modules", None))
            if targets != first_targets:
                raise ValueError(
                    "Adapters {!r} and {!r} have incompatible target_modules; "
                    "use a compatible source adapter pool".format(
                        active_names[0], name
                    )
                )
            if bool(getattr(config, "use_dora", False)):
                raise ValueError(
                    "Direct factor materialization does not support DoRA"
                )
            if getattr(config, "modules_to_save", None):
                raise ValueError(
                    "Direct materialization does not support modules_to_save"
                )
            if str(getattr(config, "bias", "none")).lower() != "none" or bool(
                getattr(config, "lora_bias", False)
            ):
                raise ValueError(
                    "Direct materialization supports factor-only adapters "
                    "with bias='none'"
                )

    def _execute_factor_plan(
        self,
        execution_plan: FusionExecutionPlan,
        temporary_name: str,
    ) -> Dict[str, int]:
        """Execute a backend-neutral plan directly into PEFT factor slices."""

        copied_modules = 0
        factor_copy_operations = 0
        factor_bytes_copied = 0
        zero_coefficient_slices = 0
        for instruction in execution_plan.modules:
            module = self._precompiled_by_key[instruction.module_key]
            lora_a = getattr(module, "lora_A", None)
            lora_b = getattr(module, "lora_B", None)
            if (
                lora_a is None
                or lora_b is None
                or not hasattr(lora_a, "__contains__")
                or not hasattr(lora_b, "__contains__")
            ):
                continue
            if temporary_name not in lora_a or temporary_name not in lora_b:
                continue

            target_a = _factor_parameter(lora_a[temporary_name])
            target_b = _factor_parameter(lora_b[temporary_name])
            found_source = False
            with _inference_no_grad(module):
                if (
                    execution_plan.factor_composition
                    is FactorComposition.LINEAR_SUM
                ):
                    target_a.zero_()
                    target_b.zero_()
                for factor_slice, a_coefficient, b_coefficient in zip(
                    execution_plan.slices,
                    instruction.a_coefficients,
                    instruction.b_coefficients,
                ):
                    source_name = factor_slice.source_name
                    if source_name not in lora_a or source_name not in lora_b:
                        continue
                    source_a = _factor_parameter(lora_a[source_name])
                    source_b = _factor_parameter(lora_b[source_name])
                    rank = int(getattr(module, "r", {}).get(source_name, 0))
                    if rank <= 0:
                        rank = int(source_a.shape[0])
                    if rank != factor_slice.rank:
                        raise ValueError(
                            "Adapter {!r} module rank {} differs from algebra "
                            "descriptor rank {}".format(
                                source_name, rank, factor_slice.rank
                            )
                        )
                    if source_a.shape[0] < rank:
                        raise ValueError(
                            "Adapter {!r} has an invalid A factor shape".format(
                                source_name
                            )
                        )
                    if source_b.shape[1] < rank:
                        raise ValueError(
                            "Adapter {!r} has an invalid B factor shape".format(
                                source_name
                            )
                        )
                    offset = factor_slice.start
                    end = factor_slice.end
                    if end > target_a.shape[0] or end > target_b.shape[1]:
                        raise RuntimeError(
                            "Temporary adapter rank is smaller than source "
                            "factor rank"
                        )
                    target_a_slice = target_a[offset:end]
                    target_b_slice = target_b[:, offset:end]
                    if (
                        execution_plan.factor_composition
                        is FactorComposition.CONCATENATE
                    ):
                        _copy_scaled_factor(
                            target_a_slice, source_a[:rank], a_coefficient
                        )
                        _copy_scaled_factor(
                            target_b_slice, source_b[:, :rank], b_coefficient
                        )
                    else:
                        _add_scaled_factor(
                            target_a_slice, source_a[:rank], a_coefficient
                        )
                        _add_scaled_factor(
                            target_b_slice, source_b[:, :rank], b_coefficient
                        )
                    if a_coefficient == 0.0 or b_coefficient == 0.0:
                        zero_coefficient_slices += 1
                    factor_copy_operations += 2
                    if a_coefficient != 0.0:
                        factor_bytes_copied += _tensor_nbytes(source_a[:rank])
                    if b_coefficient != 0.0:
                        factor_bytes_copied += _tensor_nbytes(source_b[:, :rank])
                    found_source = True

            if found_source:
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict):
                    scaling[temporary_name] = 1.0
                copied_modules += 1

        if copied_modules == 0:
            raise RuntimeError(
                "Temporary adapter was created but no LoRA factors were found"
            )
        return {
            "copied_module_count": copied_modules,
            "factor_copy_operations": factor_copy_operations,
            "factor_bytes_copied": factor_bytes_copied,
            "zero_coefficient_slices": zero_coefficient_slices,
        }


def _copy_scaled_factor(target: Any, source: Any, coefficient: float) -> None:
    """Copy one source factor into a disjoint target interval."""

    if coefficient == 0.0:
        target.zero_()
    elif coefficient == 1.0:
        target.copy_(source)
    else:
        target.copy_(source)
        target.mul_(float(coefficient))


def _add_scaled_factor(target: Any, source: Any, coefficient: float) -> None:
    """Accumulate one source factor into a shared target interval."""

    if coefficient != 0.0:
        target.add_(source, alpha=float(coefficient))


class _PeftFusedAdapter(FusedAdapterResource):
    """One candidate-scoped PEFT Adapter with independent ownership."""

    def __init__(
        self,
        adapter_key: str,
        policy: FusionPolicy,
        model: Any,
        materializer: AdapterMaterializer,
    ) -> None:
        self._adapter_key = adapter_key
        self._source_keys = active_adapter_names(policy)
        self._policy_fingerprint = policy.fingerprint()
        self._model = model
        self._materializer = materializer
        self._closed = False

    @property
    def adapter_key(self) -> str:
        return self._adapter_key

    @property
    def source_keys(self) -> Tuple[str, ...]:
        return self._source_keys

    @property
    def policy_fingerprint(self) -> str:
        return self._policy_fingerprint

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def resident_bytes(self) -> int:
        if self._closed:
            return 0
        return _peft_registered_adapter_bytes(self._model, self.adapter_key)

    @property
    def device(self) -> Optional[str]:
        if self._closed:
            return None
        try:
            return str(next(self._model.parameters()).device)
        except (AttributeError, StopIteration, TypeError):
            return None

    def close(self) -> None:
        if self._closed:
            return
        self._materializer.release(self._model, self.adapter_key)
        self._model = None
        self._closed = True


class PeftFusionAdapterFactory(FusionAdapterFactory):
    """Create PEFT fused resources using one configured fusion algebra."""

    def __init__(
        self,
        model: Optional[Any],
        materializer: AdapterMaterializer,
    ) -> None:
        self._model = model
        self.materializer = materializer

    def bind_model(self, model: Any) -> None:
        """Bind the worker-owned PEFT model without exposing mutable state."""

        if self._model is not None and self._model is not model:
            raise RuntimeError("PEFT fusion factory is bound to another model")
        self._model = model

    def create(self, policy: FusionPolicy) -> FusedAdapterResource:
        if self._model is None:
            raise RuntimeError("PEFT fusion factory is not bound to a model")
        adapter_key = "mix_candidate_{}".format(policy.fingerprint())
        self.materializer.materialize(self._model, policy, adapter_key)
        return _PeftFusedAdapter(
            adapter_key, policy, self._model, self.materializer
        )


class PeftCandidateFusion:
    """Fixed candidate-scoped fusion lifecycle for heterogeneous LoRAs.

    A temporary adapter is compiled on the first batch that uses a mixture,
    reused for the remaining benchmark samples of that candidate, and deleted
    when :meth:`candidate_session` exits. Mixed-adapter activation outside a
    candidate session is rejected so the lifecycle cannot silently degrade to
    materialize-and-release on every generation call.
    """

    def __init__(
        self,
        pool: AdapterPool,
        materializer: Optional[AdapterMaterializer] = None,
        algebra: Optional[FusionAlgebra] = None,
    ) -> None:
        self.pool = pool
        self._adapters = pool.specs
        configured_ranks = {
            adapter.name: adapter.rank or 0 for adapter in self._adapters
        }
        if isinstance(pool, ResidentAdapterPool):
            self._ranks = {
                descriptor.name: descriptor.rank
                for descriptor in pool.descriptors
            }
        else:
            self._ranks = configured_ranks
        missing_ranks = [
            adapter.name
            for adapter in self._adapters
            if int(self._ranks.get(adapter.name, 0)) <= 0
        ]
        if missing_ranks:
            raise ValueError(
                "Missing positive PEFT rank for adapter(s): {}".format(
                    ", ".join(missing_ranks)
                )
            )
        inferred_algebra = getattr(materializer, "algebra", None)
        self.algebra = algebra or inferred_algebra or create_fusion_algebra()
        self.materializer = materializer or PeftDirectAdapterMaterializer(
            pool=(pool if isinstance(pool, PeftResidentAdapterPool) else None),
            algebra=self.algebra,
        )
        self.factory = PeftFusionAdapterFactory(None, self.materializer)
        self._session_model = None
        self._session_adapters = None  # type: Optional[OrderedDict[str, FusedAdapterResource]]
        self._unreleased_adapters = OrderedDict()

    @property
    def resident_adapter_names(self) -> Tuple[str, ...]:
        names = [
            resource.adapter_key
            for resource in (self._session_adapters or {}).values()
            if not resource.closed
        ]
        names.extend(
            resource.adapter_key
            for resource in self._unreleased_adapters.values()
            if not resource.closed
        )
        return tuple(dict.fromkeys(names))

    def estimate_rank(self, mixture: FusionPolicy) -> float:
        """Report the physical rank allocated by the direct materializer.

        A scoped policy uses zeroed rank slices where an adapter is inactive,
        but one PEFT adapter still reserves the union of all adapters used in
        any scope. The proxy therefore reports the physical temporary rank.
        """

        sources = (
            self.pool.descriptors
            if isinstance(self.pool, ResidentAdapterPool)
            else tuple(
                SourceAdapterDescriptor(
                    adapter.name, self._ranks[adapter.name], 1.0
                )
                for adapter in self._adapters
            )
        )
        return float(self.algebra.estimate_rank(mixture, sources))

    @contextmanager
    def candidate_session(self, model) -> Iterator[None]:
        if self._session_adapters is not None:
            raise RuntimeError("Nested PEFT fusion candidate sessions are unsupported")
        if self._unreleased_adapters:
            self._release_resources(
                model, tuple(self._unreleased_adapters.values())
            )
        session_adapters = OrderedDict()  # type: OrderedDict[str, FusedAdapterResource]
        self._session_model = model
        self.factory.bind_model(model)
        self._session_adapters = session_adapters
        try:
            yield
        finally:
            try:
                self._release_session_adapters(model, session_adapters)
            finally:
                self._session_adapters = None
                self._session_model = None

    @contextmanager
    def activate(self, model, mixture: FusionPolicy) -> Iterator[str]:
        simple = self._simple_activation(model, mixture)
        if simple is not None:
            with simple as adapter_name:
                yield adapter_name
            return

        fingerprint = mixture.fingerprint()
        if self._session_adapters is not None:
            if self._session_model is not model:
                raise RuntimeError("Fusion session belongs to another model")
            resource = self._session_adapters.get(fingerprint)
            if resource is None:
                resource = self.factory.create(mixture)
                self._session_adapters[fingerprint] = resource
            model.set_adapter(resource.adapter_key)
            yield resource.adapter_key
            return

        raise RuntimeError(
            "Mixed-adapter inference requires backend.candidate_session(); "
            "temporary adapters are candidate-scoped"
        )

    def close(self, model) -> None:
        try:
            if self._session_adapters is not None:
                self._release_session_adapters(model, self._session_adapters)
            if self._unreleased_adapters:
                self._release_resources(
                    model, tuple(self._unreleased_adapters.values())
                )
            _restore_source_adapter(model, self._resident_source_specs())
        finally:
            self._session_adapters = None
            self._session_model = None

    def runtime_snapshot(self) -> Mapping[str, Any]:
        return {
            "implementation": type(self).__name__,
            "algebra": self.algebra.name,
            "pool": type(self.pool).__name__,
            "resident_temporary_count": len(self.resident_adapter_names),
            "unreleased_adapter_names": [
                item.adapter_key for item in self._unreleased_adapters.values()
            ],
            "materializer": dict(self.materializer.runtime_snapshot()),
        }

    @staticmethod
    def _simple_activation(model, mixture: FusionPolicy):
        uniform = uniform_policy_mixture(mixture)
        if uniform is None:
            return None
        active = uniform.active
        if not active:
            return _disable_adapters(model)
        if len(active) == 1 and abs(active[0][1] - 1.0) <= 1e-10:
            return _activate_single_adapter(model, active[0][0])
        return None

    def _release_session_adapters(
        self, model, session_adapters: OrderedDict
    ) -> None:
        resources = tuple(session_adapters.values())
        session_adapters.clear()
        for resource in resources:
            self._unreleased_adapters[resource.adapter_key] = resource
        self._release_resources(model, resources)

    def _release_resources(
        self, model, resources: Sequence[FusedAdapterResource]
    ) -> None:
        failures = []
        try:
            _restore_source_adapter(model, self._resident_source_specs())
        except Exception as exc:
            failures.append(("<source-restore>", exc))
        for resource in tuple(resources):
            try:
                resource.close()
            except Exception as exc:
                failures.append((resource.adapter_key, exc))
            else:
                self._unreleased_adapters.pop(resource.adapter_key, None)
        if failures:
            summary = "; ".join(
                "{}: {}".format(name, error) for name, error in failures
            )
            raise RuntimeError(
                "Failed to release candidate fusion resources: {}".format(summary)
            )

    def _resident_source_specs(self) -> Tuple[AdapterConfig, ...]:
        if isinstance(self.pool, ResidentAdapterPool):
            return tuple(
                self.pool.spec(name)
                for name in self.pool.resident_adapter_names
            )
        return tuple(self._adapters)


def _peft_registered_adapter_bytes(model: Any, adapter_key: str) -> int:
    total = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    for module in modules():
        for collection_name in ("lora_A", "lora_B"):
            collection = getattr(module, collection_name, None)
            if collection is None or adapter_key not in collection:
                continue
            factor = getattr(collection[adapter_key], "weight", None)
            if factor is not None:
                total += int(factor.numel()) * int(factor.element_size())
    return total


@contextmanager
def _disable_adapters(model) -> Iterator[str]:
    with model.disable_adapter():
        yield BASE_ADAPTER_NAME


@contextmanager
def _activate_single_adapter(model, adapter_name: str) -> Iterator[str]:
    model.set_adapter(adapter_name)
    yield adapter_name


def _iter_named_modules(model):
    """Yield stable paths while retaining support for lightweight test fakes."""

    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        yield from named_modules()
        return
    for index, module in enumerate(model.modules()):
        yield str(index), module


__all__ = [
    "BASE_ADAPTER_NAME",
    "AdapterMaterializer",
    "PeftDirectAdapterMaterializer",
    "PeftFusionAdapterFactory",
    "PeftCandidateFusion",
]


def _factor_parameter(value: Any):
    parameter = getattr(value, "weight", None)
    if parameter is None:
        raise TypeError("Direct materialization requires factor modules with weight")
    return parameter


def _target_module_set(value: Any):
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset((value,))
    return frozenset(str(item) for item in value)


def _peft_source_descriptor(
    name: str, config: Any
) -> SourceAdapterDescriptor:
    rank = PeftDirectAdapterMaterializer._config_rank(config)
    alpha = float(getattr(config, "lora_alpha", rank))
    native_scale = (
        alpha / math.sqrt(rank)
        if bool(getattr(config, "use_rslora", False))
        else alpha / rank
    )
    return SourceAdapterDescriptor(name, rank, native_scale)


def _temporary_lora_config(source_config: Any, total_rank: int) -> Any:
    config = deepcopy(source_config)
    config.r = total_rank
    config.lora_alpha = total_rank
    if hasattr(config, "use_rslora"):
        config.use_rslora = False
    if hasattr(config, "rank_pattern"):
        config.rank_pattern = {}
    if hasattr(config, "alpha_pattern"):
        config.alpha_pattern = {}
    if hasattr(config, "lora_dropout"):
        config.lora_dropout = 0.0
    if hasattr(config, "init_lora_weights"):
        config.init_lora_weights = False
    return config


@contextmanager
def _inference_no_grad(model) -> Iterator[None]:
    # The actual PEFT backend always has torch; avoid importing it in unit tests
    # that use lightweight fake modules.
    try:
        import torch

        with torch.no_grad():
            yield
    except ImportError:
        yield


def _restore_source_adapter(model, adapters: Sequence[AdapterConfig]) -> None:
    if adapters and hasattr(model, "set_adapter"):
        model.set_adapter(adapters[0].name)


def _adapter_is_registered(model, adapter_name: str) -> bool:
    configs = getattr(model, "peft_config", None)
    return isinstance(configs, dict) and adapter_name in configs


def _synchronize_model_device(model) -> None:
    """Synchronize CUDA at profiling boundaries; remain a no-op for fakes/CPU."""

    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _tensor_nbytes(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0
