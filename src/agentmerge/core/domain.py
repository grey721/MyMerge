"""Domain objects shared by backends, evaluators, and optimizers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class AdapterMixture:
    """A role-specific sparse mixture over a global adapter pool."""

    adapter_names: Tuple[str, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.adapter_names) != len(self.weights):
            raise ValueError("adapter_names and weights must have the same length")
        if len(set(self.adapter_names)) != len(self.adapter_names):
            raise ValueError("adapter names must be unique")
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("adapter weights must be non-negative")
        total = sum(self.weights)
        if total > 0.0 and abs(total - 1.0) > 1e-6:
            raise ValueError("non-empty adapter weights must sum to one")

    @property
    def active(self) -> Tuple[Tuple[str, float], ...]:
        return tuple(
            (name, weight)
            for name, weight in zip(self.adapter_names, self.weights)
            if weight > 0.0
        )

    @property
    def active_count(self) -> int:
        return len(self.active)

    def as_dict(self) -> Dict[str, float]:
        return dict(self.active)

    def mixture_for(self, target=None) -> "AdapterMixture":
        """Implement the FusionPolicy contract for global coefficients."""

        return self

    def iter_mixtures(self) -> Tuple["AdapterMixture", ...]:
        return (self,)

    def uniform_mixture(self) -> "AdapterMixture":
        return self

    def fingerprint(self, precision: int = 10) -> str:
        payload = [
            (name, round(weight, precision))
            for name, weight in zip(self.adapter_names, self.weights)
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]


@dataclass(frozen=True)
class GenerationRequest:
    """A backend-neutral generation request."""

    agent_name: str
    system_prompt: str
    user_prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    seed: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class AgentTrace:
    agent_name: str
    prompt: str
    # The value may be a global AdapterMixture or a scoped FusionPolicy.
    mixture: Any
    generation: GenerationResult
    parent_agents: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SystemRunResult:
    output: str
    traces: Tuple[AgentTrace, ...]
    termination_reason: str = "completed"
    steps: int = 0
    tool_error_count: int = 0

    @property
    def latency_seconds(self) -> float:
        return sum(trace.generation.latency_seconds for trace in self.traces)

    @property
    def prompt_tokens(self) -> int:
        return sum(trace.generation.prompt_tokens for trace in self.traces)

    @property
    def completion_tokens(self) -> int:
        return sum(trace.generation.completion_tokens for trace in self.traces)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class EvaluationResult:
    """Raw metrics are direction-agnostic; objectives map them to minimization."""

    metrics: Mapping[str, float]
    details: Mapping[str, Any] = field(default_factory=dict)


def average(values: Iterable[float], default: float = 0.0) -> float:
    materialized = list(values)
    if not materialized:
        return default
    return sum(materialized) / len(materialized)
