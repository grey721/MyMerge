"""PEFT-private source Adapter resources managed by the unified pool."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

from ..core.config import AdapterConfig
from .algebra import SourceAdapterDescriptor
from .pool import ResidentAdapterPool
from .resources import SourceAdapterResource


class _PeftResidentAdapter(SourceAdapterResource):
    """One independently releasable source Adapter embedded in a PEFT model."""

    def __init__(
        self,
        coordinate_index: int,
        spec: AdapterConfig,
        model: Any,
        descriptor: SourceAdapterDescriptor,
    ) -> None:
        self._coordinate_index = int(coordinate_index)
        self._spec = spec
        self._descriptor = descriptor
        self._model = model
        self._resident_bytes = _peft_adapter_bytes(model, spec.name)
        self._device = _peft_model_device(model)
        self._closed = False

    @property
    def adapter_key(self) -> str:
        return self._spec.name

    @property
    def coordinate_index(self) -> int:
        return self._coordinate_index

    @property
    def spec(self) -> AdapterConfig:
        return self._spec

    @property
    def descriptor(self) -> SourceAdapterDescriptor:
        return self._descriptor

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def resident_bytes(self) -> int:
        return 0 if self._closed else self._resident_bytes

    @property
    def device(self):
        return None if self._closed else self._device

    def close(self) -> None:
        if self._closed:
            return
        model = self._model
        configs = getattr(model, "peft_config", None)
        if isinstance(configs, dict) and self.adapter_key in configs:
            if not hasattr(model, "delete_adapter"):
                raise TypeError("PEFT model does not support Adapter deletion")
            model.delete_adapter(self.adapter_key)
        self._model = None
        self._closed = True


class PeftResidentAdapterPool(ResidentAdapterPool):
    """Own a PEFT model and one private resource object per source LoRA."""

    resource_kind = "peft_model"

    def __init__(
        self,
        adapters: Sequence[AdapterConfig],
        model: Any,
        *,
        auto_seal: bool = True,
    ) -> None:
        self._model = model
        self._temporary_cleanup_count = 0
        super().__init__((), auto_seal=False)
        for adapter in adapters:
            self.add_source(adapter)
        if auto_seal:
            self.seal()

    @classmethod
    def from_pretrained(
        cls,
        adapters: Sequence[AdapterConfig],
        base_model: Any,
        peft_model_class: Any,
    ) -> "PeftResidentAdapterPool":
        specs = tuple(adapters)
        if not specs:
            raise ValueError("PEFT requires at least one source Adapter")
        first, remaining = specs[0], specs[1:]
        model = peft_model_class.from_pretrained(
            base_model,
            first.path,
            adapter_name=first.name,
            is_trainable=False,
        )
        for adapter in remaining:
            model.load_adapter(
                adapter.path,
                adapter_name=adapter.name,
                is_trainable=False,
            )
        model.eval()
        return cls(specs, model)

    @property
    def model(self) -> Any:
        self._require_search_ready()
        return self._model

    @property
    def loaded_adapter_names(self) -> Tuple[str, ...]:
        if self.closed or self._model is None:
            return ()
        configs = getattr(self._model, "peft_config", None)
        return tuple(configs) if isinstance(configs, dict) else ()

    @property
    def temporary_adapter_names(self) -> Tuple[str, ...]:
        sources = set(self.adapter_names)
        return tuple(
            name for name in self.loaded_adapter_names if name not in sources
        )

    def _before_release(self, adapter_keys: Sequence[str]) -> None:
        if self.temporary_adapter_names:
            raise RuntimeError(
                "Cannot release PEFT source Adapters while candidate Adapters "
                "are resident"
            )
        releasing = set(adapter_keys)
        retained_names = tuple(
            adapter_key for adapter_key in self.resident_adapter_keys
            if adapter_key not in releasing
        )
        if retained_names and hasattr(self._model, "set_adapter"):
            self._model.set_adapter(retained_names[0])

    def runtime_snapshot(self) -> Mapping[str, Any]:
        snapshot = dict(super().runtime_snapshot())
        snapshot.update(
            {
                "loaded_adapter_names": list(self.loaded_adapter_names),
                "temporary_adapter_names": list(self.temporary_adapter_names),
                "temporary_cleanup_count": self._temporary_cleanup_count,
            }
        )
        return snapshot

    def close(self) -> None:
        if self.closed:
            return
        # Interrupted candidate evaluation fallback. Normal Stage-I pruning is
        # intentionally strict and rejects any live temporary Adapter.
        temporary_names = self.temporary_adapter_names
        if temporary_names and self.resident_adapter_keys and hasattr(
            self._model, "set_adapter"
        ):
            self._model.set_adapter(self.resident_adapter_keys[0])
        for name in temporary_names:
            if not hasattr(self._model, "delete_adapter"):
                raise TypeError("PEFT model does not support Adapter deletion")
            self._model.delete_adapter(name)
            self._temporary_cleanup_count += 1
        super().close()
        self._model = None

    def _create_source_resource(
        self, coordinate_index: int, adapter: AdapterConfig
    ) -> SourceAdapterResource:
        configs = getattr(self._model, "peft_config", None)
        if not isinstance(configs, dict):
            raise TypeError("PEFT model does not expose loaded Adapter configs")
        if adapter.name not in configs:
            if not hasattr(self._model, "load_adapter"):
                raise TypeError("PEFT model does not support loading Adapters")
            self._model.load_adapter(
                adapter.path,
                adapter_name=adapter.name,
                is_trainable=False,
            )
        descriptor = _inspect_peft_descriptors((adapter,), self._model)[0]
        return _PeftResidentAdapter(
            coordinate_index, adapter, self._model, descriptor
        )


def _inspect_peft_descriptors(
    adapters: Sequence[AdapterConfig], model: Any
) -> Tuple[SourceAdapterDescriptor, ...]:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict):
        raise TypeError("PEFT model does not expose loaded Adapter configs")
    descriptors = []
    for adapter in adapters:
        try:
            config = configs[adapter.name]
        except KeyError as exc:
            raise KeyError(
                "Loaded PEFT model is missing Adapter {!r}".format(adapter.name)
            ) from exc
        rank = int(getattr(config, "r", 0))
        if rank <= 0:
            raise ValueError(
                "Adapter {!r} has invalid PEFT rank {}".format(
                    adapter.name, rank
                )
            )
        if adapter.rank is not None and int(adapter.rank) != rank:
            raise ValueError(
                "Adapter {!r} configured rank {} but loaded PEFT rank is {}"
                .format(adapter.name, adapter.rank, rank)
            )
        if getattr(config, "rank_pattern", None):
            raise ValueError("PEFT resident pool requires uniform rank")
        if getattr(config, "alpha_pattern", None):
            raise ValueError("PEFT resident pool requires uniform alpha")
        alpha = float(getattr(config, "lora_alpha", rank))
        native_scale = (
            alpha / math.sqrt(rank)
            if bool(getattr(config, "use_rslora", False))
            else alpha / rank
        )
        descriptors.append(
            SourceAdapterDescriptor(adapter.name, rank, native_scale)
        )
    return tuple(descriptors)


def _peft_adapter_bytes(model: Any, adapter_name: str) -> int:
    total = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return 0
    for module in modules():
        for collection_name in ("lora_A", "lora_B"):
            collection = getattr(module, collection_name, None)
            if collection is None or adapter_name not in collection:
                continue
            value = collection[adapter_name]
            tensor = getattr(value, "weight", value)
            numel = getattr(tensor, "numel", None)
            element_size = getattr(tensor, "element_size", None)
            if callable(numel) and callable(element_size):
                total += int(numel()) * int(element_size())
    return total


def _peft_model_device(model: Any):
    try:
        return str(model.get_input_embeddings().weight.device)
    except (AttributeError, StopIteration, TypeError):
        try:
            return str(next(model.parameters()).device)
        except (AttributeError, StopIteration, TypeError):
            return None


__all__ = ["PeftResidentAdapterPool"]
