"""Backend abstraction for model inference with role-specific adapter mixtures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Mapping, Sequence, Tuple

from ..core.config import AdapterConfig
from ..core.domain import GenerationRequest, GenerationResult
from ..core.policies import FusionPolicy, iter_policy_mixtures


class OptionalDependencyError(RuntimeError):
    """Raised when a selected backend dependency is not installed."""


class ModelBackend(ABC):
    """A model runtime capable of applying one adapter mixture per request."""

    @property
    def adapter_pool(self):
        """Expose the backend's unified Adapter resource owner."""

        pool = getattr(self, "_pool", None)
        if pool is None:
            raise RuntimeError(
                "{} does not expose an AdapterPool".format(type(self).__name__)
            )
        return pool

    @property
    @abstractmethod
    def adapter_names(self) -> Sequence[str]:
        """Return the global adapter pool in stable order."""

    @property
    @abstractmethod
    def adapter_specs(self) -> Sequence[AdapterConfig]:
        """Return immutable metadata for cost estimation and provenance."""

    @abstractmethod
    def generate(
        self, request: GenerationRequest, mixture: FusionPolicy
    ) -> GenerationResult:
        """Generate one agent response under the supplied adapter mixture."""

    def generate_many(
        self,
        requests: Sequence[GenerationRequest],
        mixtures: Sequence[FusionPolicy],
    ) -> Tuple[GenerationResult, ...]:
        """Generate a batch while preserving request order.

        The default implementation is deliberately conservative and delegates
        to :meth:`generate`. Device-aware backends should override this method
        to perform one batched model call. Allowing one mixture per request also
        keeps the contract usable by future serving backends that can route
        different LoRAs within the same batch.
        """

        if len(requests) != len(mixtures):
            raise ValueError("requests and mixtures must have the same length")
        return tuple(
            self.generate(request, mixture)
            for request, mixture in zip(requests, mixtures)
        )

    def count_prompt_tokens(self, request: GenerationRequest) -> int:
        """Count prompt tokens for backend-neutral loop context guards.

        Concrete tokenizer-backed runtimes override this method.  The fallback
        keeps Mock and third-party backends usable without imposing a tokenizer
        interface on the optimizer or Agent system.
        """

        text = request.system_prompt + "\n" + request.user_prompt
        return max(1, len(text.split()))

    @contextmanager
    def candidate_session(self):
        """Scope backend resources to one complete genome evaluation.

        PEFT backends use this hook to lazily compile one temporary adapter per
        distinct Agent mixture and reuse it across all benchmark micro-batches.
        Backends without mutable fusion state keep the no-op implementation.
        """

        yield

    def estimate_fusion_rank(self, mixture: FusionPolicy) -> float:
        """Estimate effective LoRA rank; backends may refine this proxy."""

        ranks = {
            adapter.name: adapter.rank or 0 for adapter in self.adapter_specs
        }
        estimates = [
            float(sum(ranks[name] for name, _ in local.active))
            for local in iter_policy_mixtures(mixture)
        ]
        return sum(estimates) / len(estimates) if estimates else 0.0

    def runtime_snapshot(self) -> Mapping[str, Any]:
        """Return lightweight, JSON-serializable backend diagnostics.

        The search stack does not depend on these fields. Runtime-specific
        smoke tests and profilers can use this public hook without reaching
        into a backend's private model or fusion implementation.
        """

        snapshot = {
            "backend": type(self).__name__,
            "source_adapter_names": list(self.adapter_names),
        }
        pool = getattr(self, "_pool", None)
        if pool is not None:
            snapshot["adapter_pool"] = dict(pool.runtime_snapshot())
        return snapshot

    def release_unreferenced_adapters(
        self, reference_counts: Mapping[str, int]
    ) -> Mapping[str, Any]:
        """Release backend resources for globally unreferenced Adapter keys."""

        return dict(
            self.adapter_pool.release_unreferenced(reference_counts).as_dict()
        )

    def close(self) -> None:
        """Release runtime resources. Backends may override this hook."""

    def __enter__(self) -> "ModelBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
