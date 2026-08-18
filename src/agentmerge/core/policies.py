"""Backend-neutral fusion plans and module-granularity policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Optional, Protocol, Tuple

from .domain import AdapterMixture


@dataclass(frozen=True)
class ModuleTarget:
    """Backend-neutral identity of one adapter-bearing model module."""

    module_path: str = ""
    layer_index: Optional[int] = None
    module_type: Optional[str] = None


GLOBAL_TARGET = ModuleTarget()


class ModuleLocator(ABC):
    @abstractmethod
    def locate(self, module_path: str, module: object = None) -> ModuleTarget:
        pass


class TransformerModuleLocator(ModuleLocator):
    """Resolve common Hugging Face Transformer module paths."""

    _LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

    def locate(self, module_path: str, module: object = None) -> ModuleTarget:
        match = self._LAYER_PATTERN.search(module_path)
        layer_index = int(match.group(1)) if match is not None else None
        module_type = module_path.rsplit(".", 1)[-1] if module_path else None
        return ModuleTarget(
            module_path=module_path,
            layer_index=layer_index,
            module_type=module_type,
        )


class FusionPolicy(Protocol):
    """A per-Agent runtime policy queried at arbitrary model targets."""

    adapter_names: Tuple[str, ...]

    def mixture_for(self, target: ModuleTarget = GLOBAL_TARGET) -> AdapterMixture:
        ...

    def iter_mixtures(self) -> Tuple[AdapterMixture, ...]:
        ...

    def fingerprint(self, precision: int = 10) -> str:
        ...

    def as_dict(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class FusionPlan:
    """Evaluated phenotype: one backend-neutral fusion policy per Agent.

    Optimizers may use unrelated chromosomes, codecs, repair operators, and
    granularities. Evaluation and runtime code depend only on this immutable
    plan, which is also safe to send to persistent GPU worker processes.
    """

    agent_policies: Tuple[Tuple[str, FusionPolicy], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.agent_policies)
        if not names:
            raise ValueError("FusionPlan requires at least one Agent policy")
        if len(set(names)) != len(names):
            raise ValueError("FusionPlan Agent names must be unique")

    @property
    def agent_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.agent_policies)

    def policy_for(self, agent_name: str) -> FusionPolicy:
        for name, policy in self.agent_policies:
            if name == agent_name:
                return policy
        raise KeyError("FusionPlan has no Agent {!r}".format(agent_name))

    def fingerprint(self, precision: int = 10) -> str:
        payload = [
            [name, policy.fingerprint(precision)]
            for name, policy in self.agent_policies
        ]
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> Mapping[str, Any]:
        return {
            name: dict(policy.as_dict())
            for name, policy in self.agent_policies
        }


class FusionGranularity(ABC):
    """Map a model target to a stable coefficient scope."""

    @property
    @abstractmethod
    def scope_names(self) -> Tuple[str, ...]:
        pass

    @abstractmethod
    def scope_for(self, target: ModuleTarget) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> Mapping[str, Any]:
        pass


@dataclass(frozen=True)
class GlobalGranularity(FusionGranularity):
    @property
    def scope_names(self) -> Tuple[str, ...]:
        return ("global",)

    def scope_for(self, target: ModuleTarget) -> str:
        return "global"

    def to_dict(self) -> Mapping[str, Any]:
        return {"type": "global"}


@dataclass(frozen=True)
class LayerGranularity(FusionGranularity):
    num_layers: int

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple("layer:{}".format(index) for index in range(self.num_layers))

    def scope_for(self, target: ModuleTarget) -> str:
        index = _require_layer(target, self.num_layers)
        return "layer:{}".format(index)

    def to_dict(self) -> Mapping[str, Any]:
        return {"type": "layer", "num_layers": self.num_layers}


@dataclass(frozen=True)
class LayerGroupGranularity(FusionGranularity):
    num_layers: int
    group_size: int

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")

    @property
    def group_count(self) -> int:
        return (self.num_layers + self.group_size - 1) // self.group_size

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple("group:{}".format(index) for index in range(self.group_count))

    def scope_for(self, target: ModuleTarget) -> str:
        index = _require_layer(target, self.num_layers)
        return "group:{}".format(index // self.group_size)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": "layer_group",
            "num_layers": self.num_layers,
            "group_size": self.group_size,
        }


@dataclass(frozen=True)
class ModuleTypeGranularity(FusionGranularity):
    module_types: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.module_types:
            raise ValueError("module_types cannot be empty")
        if len(set(self.module_types)) != len(self.module_types):
            raise ValueError("module_types must be unique")

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple("module:{}".format(name) for name in self.module_types)

    def scope_for(self, target: ModuleTarget) -> str:
        if target.module_type not in self.module_types:
            raise ValueError(
                "Unknown module type {!r}; configured: {}".format(
                    target.module_type, ", ".join(self.module_types)
                )
            )
        return "module:{}".format(target.module_type)

    def to_dict(self) -> Mapping[str, Any]:
        return {"type": "module_type", "module_types": list(self.module_types)}


@dataclass(frozen=True)
class LayerModuleGranularity(FusionGranularity):
    num_layers: int
    module_types: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not self.module_types:
            raise ValueError("module_types cannot be empty")
        if len(set(self.module_types)) != len(self.module_types):
            raise ValueError("module_types must be unique")

    @property
    def scope_names(self) -> Tuple[str, ...]:
        return tuple(
            "layer:{}:module:{}".format(layer, module_type)
            for layer in range(self.num_layers)
            for module_type in self.module_types
        )

    def scope_for(self, target: ModuleTarget) -> str:
        layer = _require_layer(target, self.num_layers)
        if target.module_type not in self.module_types:
            raise ValueError(
                "Unknown module type {!r}; configured: {}".format(
                    target.module_type, ", ".join(self.module_types)
                )
            )
        return "layer:{}:module:{}".format(layer, target.module_type)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "type": "layer_module",
            "num_layers": self.num_layers,
            "module_types": list(self.module_types),
        }


@dataclass(frozen=True)
class ScopedAdapterFusionPolicy:
    """Immutable coefficient table over arbitrary granularity scopes."""

    adapter_names: Tuple[str, ...]
    granularity: FusionGranularity
    scope_weights: Tuple[Tuple[str, Tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        if not self.adapter_names:
            raise ValueError("adapter_names cannot be empty")
        if len(set(self.adapter_names)) != len(self.adapter_names):
            raise ValueError("adapter_names must be unique")
        names = tuple(name for name, _ in self.scope_weights)
        if names != self.granularity.scope_names:
            raise ValueError(
                "scope_weights must follow granularity scope order: {} != {}".format(
                    names, self.granularity.scope_names
                )
            )
        for scope_name, weights in self.scope_weights:
            AdapterMixture(self.adapter_names, weights)

    def mixture_for(self, target: ModuleTarget = GLOBAL_TARGET) -> AdapterMixture:
        scope = self.granularity.scope_for(target)
        for scope_name, weights in self.scope_weights:
            if scope_name == scope:
                return AdapterMixture(self.adapter_names, weights)
        raise KeyError("No coefficients configured for scope {!r}".format(scope))

    def iter_mixtures(self) -> Tuple[AdapterMixture, ...]:
        return tuple(
            AdapterMixture(self.adapter_names, weights)
            for _, weights in self.scope_weights
        )

    def uniform_mixture(self) -> Optional[AdapterMixture]:
        mixtures = self.iter_mixtures()
        if not mixtures:
            return None
        first = mixtures[0]
        return first if all(item == first for item in mixtures[1:]) else None

    @property
    def active_count(self) -> int:
        return len(active_adapter_names(self))

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "granularity": dict(self.granularity.to_dict()),
            "scopes": {
                scope: {
                    name: weight
                    for name, weight in zip(self.adapter_names, weights)
                    if weight > 0.0
                }
                for scope, weights in self.scope_weights
            },
        }

    def fingerprint(self, precision: int = 10) -> str:
        payload = {
            "adapter_names": list(self.adapter_names),
            "granularity": dict(self.granularity.to_dict()),
            "scope_weights": [
                [scope, [round(value, precision) for value in weights]]
                for scope, weights in self.scope_weights
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:16]


def iter_policy_mixtures(policy: FusionPolicy) -> Tuple[AdapterMixture, ...]:
    method = getattr(policy, "iter_mixtures", None)
    if method is not None:
        return tuple(method())
    if isinstance(policy, AdapterMixture):
        return (policy,)
    raise TypeError("Object does not implement FusionPolicy")


def uniform_policy_mixture(policy: FusionPolicy) -> Optional[AdapterMixture]:
    method = getattr(policy, "uniform_mixture", None)
    if method is not None:
        return method()
    mixtures = iter_policy_mixtures(policy)
    first = mixtures[0]
    return first if all(item == first for item in mixtures[1:]) else None


def active_adapter_names(policy: FusionPolicy) -> Tuple[str, ...]:
    active = set()
    for mixture in iter_policy_mixtures(policy):
        active.update(name for name, _ in mixture.active)
    return tuple(name for name in policy.adapter_names if name in active)


def average_policy_mixture(policy: FusionPolicy) -> AdapterMixture:
    mixtures = iter_policy_mixtures(policy)
    count = len(mixtures)
    weights = tuple(
        sum(mixture.weights[index] for mixture in mixtures) / count
        for index in range(len(policy.adapter_names))
    )
    total = sum(weights)
    if total > 0.0:
        weights = tuple(value / total for value in weights)
    return AdapterMixture(tuple(policy.adapter_names), weights)


def _require_layer(target: ModuleTarget, num_layers: int) -> int:
    if target.layer_index is None:
        raise ValueError(
            "Fusion granularity requires a layer index for module {!r}".format(
                target.module_path
            )
        )
    if not 0 <= target.layer_index < num_layers:
        raise ValueError(
            "Layer index {} outside configured range 0..{}".format(
                target.layer_index, num_layers - 1
            )
        )
    return target.layer_index


__all__ = [
    "GLOBAL_TARGET",
    "FusionGranularity",
    "FusionPlan",
    "FusionPolicy",
    "GlobalGranularity",
    "LayerGranularity",
    "LayerGroupGranularity",
    "LayerModuleGranularity",
    "ModuleLocator",
    "ModuleTarget",
    "ModuleTypeGranularity",
    "ScopedAdapterFusionPolicy",
    "TransformerModuleLocator",
    "active_adapter_names",
    "average_policy_mixture",
    "iter_policy_mixtures",
    "uniform_policy_mixture",
]
