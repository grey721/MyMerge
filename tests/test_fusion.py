import random

from agentmerge.optimizers.genome import FusionGenome
from agentmerge.optimizers.representation import FusionSearchSpace


def test_sample_and_repair_preserve_sparse_simplex():
    rng = random.Random(7)
    space = FusionSearchSpace(
        agent_names=("a", "b"),
        adapter_names=("x", "y", "z", "w"),
        min_active=1,
        max_active=3,
    )
    for _ in range(100):
        genome = space.sample(rng)
        for vector in genome.agent_weights.values():
            assert abs(sum(vector) - 1.0) < 1e-9
            assert 1 <= sum(value > 0 for value in vector) <= 3

    invalid = FusionGenome({"a": [-1, 2, 3, 4], "b": [0, 0, 0, 0]})
    repaired = space.repair(invalid, rng)
    for vector in repaired.agent_weights.values():
        assert all(value >= 0 for value in vector)
        assert abs(sum(vector) - 1.0) < 1e-9
        assert 1 <= sum(value > 0 for value in vector) <= 3
