"""Real-backend smoke diagnostics independent of the search algorithm."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..backends import ModelBackend, create_model_backend
from ..backends.materialization import BASE_ADAPTER_NAME
from ..core.config import ExperimentConfig
from ..core.domain import AdapterMixture, GenerationRequest, GenerationResult


@dataclass(frozen=True)
class SmokeGeneration:
    label: str
    adapter: str
    mixture: Mapping[str, float]
    text: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    activation_latency_seconds: float
    model_generation_latency_seconds: float
    fusion_cleanup_latency_seconds: float


@dataclass(frozen=True)
class BackendSmokeReport:
    passed: bool
    backend_load_seconds: float
    generations: Tuple[SmokeGeneration, ...]
    lifecycle_checks: Mapping[str, bool]
    after_load: Mapping[str, Any]
    during_fusion: Mapping[str, Any]
    after_candidate_session: Mapping[str, Any]
    candidate_session_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackendSmokeRunner:
    """Exercise base, one-hot and candidate-scoped fused inference.

    This runner owns the backend for the duration of one diagnostic run. It
    intentionally bypasses the Agent DAG and optimizer so failures can be
    attributed to model loading, source adapters, generation, or fusion.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        backend_factory: Optional[Callable[[], ModelBackend]] = None,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory or self._create_backend

    def run(
        self,
        prompt: str = "What is 2 + 2? Reply with only the answer.",
        max_new_tokens: int = 16,
        one_hot_limit: int = 1,
    ) -> BackendSmokeReport:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if one_hot_limit < 1:
            raise ValueError("one_hot_limit must be positive")
        if len(self.config.adapters) < 2:
            raise ValueError("Fusion smoke test requires at least two adapters")
        load_start = time.perf_counter()
        backend = self._backend_factory()
        load_seconds = time.perf_counter() - load_start
        try:
            names = tuple(backend.adapter_names)
            after_load = dict(backend.runtime_snapshot())
            generations = []  # type: List[SmokeGeneration]

            base = AdapterMixture(names, tuple(0.0 for _ in names))
            base_result = backend.generate(
                self._request(prompt, max_new_tokens), base
            )
            generations.append(self._generation("base", base, base_result))

            for index, adapter_name in enumerate(names[:one_hot_limit]):
                weights = tuple(
                    1.0 if item_index == index else 0.0
                    for item_index in range(len(names))
                )
                mixture = AdapterMixture(names, weights)
                result = backend.generate(
                    self._request(prompt, max_new_tokens), mixture
                )
                generations.append(
                    self._generation(
                        "one_hot:{}".format(adapter_name), mixture, result
                    )
                )

            dual_weights = (0.5, 0.5) + tuple(
                0.0 for _ in range(len(names) - 2)
            )
            dual = AdapterMixture(names, dual_weights)
            session_start = time.perf_counter()
            with backend.candidate_session():
                first_fused = backend.generate(
                    self._request(prompt, max_new_tokens), dual
                )
                during_fusion = dict(backend.runtime_snapshot())
                second_fused = backend.generate(
                    self._request(prompt, max_new_tokens), dual
                )
                after_reuse = dict(backend.runtime_snapshot())
            candidate_session_seconds = time.perf_counter() - session_start
            after_session = dict(backend.runtime_snapshot())

            generations.append(
                self._generation("fusion:first", dual, first_fused)
            )
            generations.append(
                self._generation("fusion:reused", dual, second_fused)
            )
            lifecycle = self._lifecycle_checks(
                names=names,
                base_result=base_result,
                first_fused=first_fused,
                second_fused=second_fused,
                during_fusion=during_fusion,
                after_reuse=after_reuse,
                after_session=after_session,
            )
            return BackendSmokeReport(
                passed=all(lifecycle.values()),
                backend_load_seconds=load_seconds,
                generations=tuple(generations),
                lifecycle_checks=lifecycle,
                after_load=after_load,
                during_fusion=during_fusion,
                after_candidate_session=after_session,
                candidate_session_seconds=candidate_session_seconds,
            )
        finally:
            backend.close()

    def _create_backend(self) -> ModelBackend:
        if self.config.model.backend.lower() not in ("peft", "vllm"):
            raise ValueError(
                "Real backend smoke test requires model.backend=peft or vllm"
            )
        return create_model_backend(self.config)

    def _request(self, prompt: str, max_new_tokens: int) -> GenerationRequest:
        return GenerationRequest(
            agent_name="backend_smoke",
            system_prompt="Answer the user directly and concisely.",
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            seed=self.config.model.generation.seed or self.config.seed,
        )

    @staticmethod
    def _generation(
        label: str,
        mixture: AdapterMixture,
        result: GenerationResult,
    ) -> SmokeGeneration:
        return SmokeGeneration(
            label=label,
            adapter=str(result.metadata.get("adapter", "")),
            mixture=mixture.as_dict(),
            text=result.text[:500],
            latency_seconds=result.latency_seconds,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            activation_latency_seconds=float(
                result.metadata.get("activation_latency_seconds", 0.0)
            ),
            model_generation_latency_seconds=float(
                result.metadata.get("model_generation_latency_seconds", 0.0)
            ),
            fusion_cleanup_latency_seconds=float(
                result.metadata.get("fusion_cleanup_latency_seconds", 0.0)
            ),
        )

    @staticmethod
    def _lifecycle_checks(
        names: Tuple[str, ...],
        base_result: GenerationResult,
        first_fused: GenerationResult,
        second_fused: GenerationResult,
        during_fusion: Mapping[str, Any],
        after_reuse: Mapping[str, Any],
        after_session: Mapping[str, Any],
    ) -> Mapping[str, bool]:
        first_name = str(first_fused.metadata.get("adapter", ""))
        second_name = str(second_fused.metadata.get("adapter", ""))
        resident_during = tuple(
            during_fusion.get("candidate_adapter_names", ())
        )
        resident_after_reuse = tuple(
            after_reuse.get("candidate_adapter_names", ())
        )
        loaded_after = tuple(after_session.get("loaded_adapter_names", ()))
        temporary_after = tuple(
            after_session.get("temporary_adapter_names", ())
        )
        return {
            "base_adapter_disabled": (
                base_result.metadata.get("adapter") == BASE_ADAPTER_NAME
            ),
            "fusion_materialized_once": (
                len(resident_during) == 1 and resident_during[0] == first_name
            ),
            "fusion_reused_within_candidate": (
                bool(first_name)
                and first_name == second_name
                and resident_during == resident_after_reuse
            ),
            "fusion_released_after_candidate": (
                not after_session.get("candidate_adapter_names")
                and not temporary_after
            ),
            "source_adapters_preserved": set(loaded_after) == set(names),
        }


__all__ = [
    "BackendSmokeReport",
    "BackendSmokeRunner",
    "SmokeGeneration",
]
