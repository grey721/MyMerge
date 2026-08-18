"""Engine-neutral transport for AgentMerge in-memory vLLM LoRAs.

The fusion layer owns tensors and Adapter lifetimes.  This module owns the
small, version-scoped boundary used to register those tensors with either the
vLLM V0 or V1 offline engine.  V1 is deliberately kept in-process: CUDA
tensors never cross EngineCore IPC, while inference still uses the V1
scheduler and model runner.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Type

from .tensor_lora import install_tensor_lora_support, make_tensor_lora_request


SUPPORTED_ENGINE_MODES = ("v0", "v1")


def normalize_engine_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in SUPPORTED_ENGINE_MODES:
        raise ValueError(
            "vllm.engine_mode must be one of {}; got {!r}".format(
                ", ".join(SUPPORTED_ENGINE_MODES), value
            )
        )
    return mode


def configure_vllm_engine_mode(value: str) -> str:
    """Select V0/V1 before importing vLLM and force tensor-safe locality."""

    mode = normalize_engine_mode(value)
    os.environ["VLLM_USE_V1"] = "1" if mode == "v1" else "0"
    # AgentMerge's outer runtime already gives every backend process one GPU.
    # Keeping the offline EngineCore in that process lets LoRA requests retain
    # direct references to resident CUDA tensors.  V1 scheduler/model-runner
    # behavior remains enabled; only the optional EngineCore process split is
    # disabled.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return mode


class VllmTensorLoraTransport:
    """Register and release tensor-backed LoRAs through one vLLM engine.

    The same public engine operations exist in vLLM 0.10 V0 and V1.  Keeping
    them behind this object prevents AdapterPool and fusion materialization
    from depending on either engine implementation.
    """

    name = "inproc_tensor"

    def __init__(
        self,
        engine_mode: str,
        request_class: Type[Any],
        *,
        load_device: str = "cuda",
    ) -> None:
        self._engine_mode = normalize_engine_mode(engine_mode)
        self._request_class = request_class
        self._load_device = str(load_device)
        self._engine = None  # type: Optional[Any]
        self.registration_count = 0
        self.unregistration_count = 0

    @classmethod
    def prepare(
        cls, engine_mode: str = "v1", *, load_device: str = "cuda"
    ) -> "VllmTensorLoraTransport":
        """Configure vLLM, install tensor loading, then await engine binding."""

        mode = configure_vllm_engine_mode(engine_mode)
        request_class = install_tensor_lora_support()
        return cls(mode, request_class, load_device=load_device)

    @property
    def engine_mode(self) -> str:
        return self._engine_mode

    @property
    def display_engine_mode(self) -> str:
        return self._engine_mode.upper()

    @property
    def load_device(self) -> str:
        return self._load_device

    @property
    def bound(self) -> bool:
        return self._engine is not None

    def bind_engine(self, engine: Any) -> None:
        if self._engine is not None:
            if self._engine is engine:
                return
            raise RuntimeError("vLLM LoRA transport is already bound")
        self._validate_engine_mode(engine)
        self._engine = engine

    def make_request(
        self,
        *,
        name: str,
        request_id: int,
        peft_config: Mapping[str, Any],
        tensors: Mapping[str, Any],
    ) -> Any:
        return make_tensor_lora_request(
            self._request_class,
            name=name,
            request_id=request_id,
            peft_config=peft_config,
            tensors=tensors,
            load_device=self._load_device,
        )

    def register(self, request: Any) -> bool:
        engine = self._require_engine()
        result = engine.add_lora(request)
        self.registration_count += 1
        return result is not False

    def unregister(self, request_id: int) -> bool:
        engine = self._require_engine()
        result = engine.remove_lora(int(request_id))
        self.unregistration_count += 1
        return result is not False

    def list_adapter_ids(self):
        engine = self._require_engine()
        if not hasattr(engine, "list_loras"):
            raise RuntimeError("The bound vLLM engine cannot list LoRAs")
        return set(engine.list_loras())

    def runtime_snapshot(self):
        return {
            "implementation": type(self).__name__,
            "name": self.name,
            "engine_mode": self.display_engine_mode,
            "process_boundary": "in_process",
            "load_device": self._load_device,
            "bound": self.bound,
            "registration_count": self.registration_count,
            "unregistration_count": self.unregistration_count,
        }

    def close(self) -> None:
        self._engine = None

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("vLLM LoRA transport is not bound")
        return self._engine

    def _validate_engine_mode(self, engine: Any) -> None:
        module = type(engine).__module__
        if not module.startswith("vllm."):
            return
        actual = "v1" if module.startswith("vllm.v1.") else "v0"
        if actual != self._engine_mode:
            raise RuntimeError(
                "Configured vLLM {} transport received a {} engine ({})".format(
                    self.display_engine_mode, actual.upper(), module
                )
            )


__all__ = [
    "SUPPORTED_ENGINE_MODES",
    "VllmTensorLoraTransport",
    "configure_vllm_engine_mode",
    "normalize_engine_mode",
]
