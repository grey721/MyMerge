"""Backend construction registry kept outside experiment orchestration."""

from __future__ import annotations

from ..core.config import ExperimentConfig
from .base import ModelBackend
from .mock import MockModelBackend
from .peft import PeftCausalLMBackend
from .vllm import VllmCausalLMBackend


def create_model_backend(config: ExperimentConfig) -> ModelBackend:
    backend_name = config.model.backend.lower()
    if backend_name == "mock":
        return MockModelBackend(config.adapters)
    if backend_name == "peft":
        return PeftCausalLMBackend(config.model, config.adapters)
    if backend_name == "vllm":
        return VllmCausalLMBackend(
            config.model,
            config.adapters,
            candidate_lora_capacity=len(config.agents),
            inference_batch_size=config.benchmark.inference_batch_size,
        )
    raise ValueError("Unknown model backend {!r}".format(backend_name))


__all__ = ["create_model_backend"]
