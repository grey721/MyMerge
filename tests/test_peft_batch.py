from contextlib import nullcontext
import threading

from agentmerge.backends.materialization import PeftCandidateFusion
from agentmerge.backends.peft import PeftCausalLMBackend
from agentmerge.backends.pool import AdapterPool
from agentmerge.core.config import AdapterConfig, GenerationConfig, ModelConfig
from agentmerge.core.domain import AdapterMixture, GenerationRequest


class _Vector:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class _Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def to(self, device):
        return self

    def sum(self, dim):
        assert dim == -1
        return _Vector(sum(row) for row in self.rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_slice, column_slice = key
            assert row_slice == slice(None)
            return _Matrix(row[column_slice] for row in self.rows)
        return _Vector(self.rows[key])


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def __call__(self, prompts, return_tensors, padding):
        tokenized = [list(range(1, len(prompt.split()) + 1)) for prompt in prompts]
        width = max(len(tokens) for tokens in tokenized)
        input_ids = [[0] * (width - len(tokens)) + tokens for tokens in tokenized]
        attention = [
            [0] * (width - len(tokens)) + [1] * len(tokens)
            for tokens in tokenized
        ]
        return {
            "input_ids": _Matrix(input_ids),
            "attention_mask": _Matrix(attention),
        }

    def decode(self, token_ids, skip_special_tokens):
        return " ".join(
            str(value) for value in token_ids.tolist() if value != self.pad_token_id
        )


class _Cuda:
    @staticmethod
    def is_available():
        return False


class _Torch:
    cuda = _Cuda()

    @staticmethod
    def manual_seed(seed):
        return None

    @staticmethod
    def inference_mode():
        return nullcontext()


class _Weight:
    device = "cpu"


class _Embeddings:
    weight = _Weight()


class _Model:
    def __init__(self):
        self.batch_sizes = []
        self.final_adapter = None

    def get_input_embeddings(self):
        return _Embeddings()

    def set_adapter(self, name):
        self.final_adapter = name

    def generate(self, input_ids, attention_mask, **kwargs):
        self.batch_sizes.append(input_ids.shape[0])
        return _Matrix(
            row + [90 + index, 0]
            for index, row in enumerate(input_ids.rows)
        )


class _Pool(AdapterPool):
    def __init__(self, adapters, model):
        super().__init__(adapters)
        self.model = model


def _request(prompt):
    return GenerationRequest(
        agent_name="agent",
        system_prompt="system",
        user_prompt=prompt,
        max_new_tokens=8,
        temperature=0.0,
        seed=7,
    )


def test_peft_generate_many_groups_compatible_mixtures_and_preserves_order():
    adapters = (
        AdapterConfig("a", "a", rank=8),
        AdapterConfig("b", "b", rank=16),
    )
    backend = object.__new__(PeftCausalLMBackend)
    backend.model_config = ModelConfig(
        backend="peft",
        name_or_path="fake",
        generation=GenerationConfig(),
    )
    backend._lock = threading.RLock()
    backend._torch = _Torch()
    backend._tokenizer = _Tokenizer()
    backend._pool = _Pool(adapters, _Model())
    backend._candidate_fusion = PeftCandidateFusion(backend._pool)

    requests = (
        _request("short"),
        _request("a longer prompt"),
        _request("two words"),
    )
    mixture_a = AdapterMixture(("a", "b"), (1.0, 0.0))
    mixture_b = AdapterMixture(("a", "b"), (0.0, 1.0))
    results = backend.generate_many(
        requests, (mixture_a, mixture_b, mixture_a)
    )

    assert backend._model.batch_sizes == [2, 1]
    assert [result.metadata["adapter"] for result in results] == ["a", "b", "a"]
    assert [result.prompt_tokens for result in results] == [1, 3, 2]
    assert [result.completion_tokens for result in results] == [1, 1, 1]
    assert [result.metadata["batch_size"] for result in results] == [2, 1, 2]
