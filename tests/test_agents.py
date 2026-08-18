from agentmerge.evaluation.system import (
    FixedTopology,
    MultiAgentSystem,
    ToolEnvironment,
    ToolLoopSystem,
    ToolLoopTopology,
)
from agentmerge.backends import MockModelBackend
from agentmerge.core.config import (
    AdapterConfig,
    AgentConfig,
    GenerationConfig,
    TopologyConfig,
)
from agentmerge.core.domain import (
    AdapterMixture,
    GenerationResult,
)
from agentmerge.core.policies import FusionPlan


def _plan(first, last):
    return FusionPlan(
        (
            ("first", AdapterMixture(("a", "b"), tuple(first))),
            ("last", AdapterMixture(("a", "b"), tuple(last))),
        )
    )


def test_fixed_topology_passes_upstream_messages():
    adapters = (
        AdapterConfig(name="a", path="a"),
        AdapterConfig(name="b", path="b"),
    )
    agents = (
        AgentConfig("first", "planner", "plan"),
        AgentConfig("last", "critic", "final"),
    )
    topology = FixedTopology(
        ("first", "last"),
        TopologyConfig(
            edges=(("first", "last"),),
            output_agent="last",
            kind="fixed",
        ),
    )
    system = MultiAgentSystem(
        MockModelBackend(adapters), agents, topology, GenerationConfig()
    )
    plan = _plan((1.0, 0.0), (0.0, 1.0))
    result = system.run("Compute [expected: 42]", plan)
    assert result.output == "42"
    assert result.traces[1].parent_agents == ("first",)
    assert "[first]" in result.traces[1].prompt
    assert result.traces[0].mixture.as_dict() == {"a": 1.0}


class _RecordingBatchBackend(MockModelBackend):
    def __init__(self, adapters):
        super().__init__(adapters)
        self.batch_sizes = []

    def generate_many(self, requests, mixtures):
        self.batch_sizes.append(len(requests))
        return super().generate_many(requests, mixtures)


def test_run_many_batches_tasks_by_topology_stage():
    adapters = (
        AdapterConfig(name="a", path="a"),
        AdapterConfig(name="b", path="b"),
    )
    agents = (
        AgentConfig("first", "planner", "plan"),
        AgentConfig("last", "critic", "final"),
    )
    topology = FixedTopology(
        ("first", "last"),
        TopologyConfig(
            edges=(("first", "last"),),
            output_agent="last",
            kind="fixed",
        ),
    )
    backend = _RecordingBatchBackend(adapters)
    system = MultiAgentSystem(
        backend, agents, topology, GenerationConfig(seed=11)
    )
    plan = _plan((1.0, 0.0), (0.0, 1.0))

    results = system.run_many(
        ("Compute [expected: 42]", "Compute [expected: 7]"),
        plan,
        ({"case_id": "one"}, {"case_id": "two"}),
    )

    assert backend.batch_sizes == [2, 2]
    assert [result.output for result in results] == ["42", "7"]
    assert all(len(result.traces) == 2 for result in results)
    assert "[first]" in results[0].traces[1].prompt


class _LoopBackend(MockModelBackend):
    def __init__(self, adapters):
        super().__init__(adapters)
        self.batch_sizes = []

    def generate_many(self, requests, mixtures):
        self.batch_sizes.append(len(requests))
        results = []
        for request, mixture in zip(requests, mixtures):
            self._pool.validate_policy(mixture)
            if request.agent_name == "planner":
                text = "lookup"
            elif request.agent_name == "arguments":
                text = "query-{}".format(request.metadata["case_id"])
            else:
                text = "final-{}".format(request.metadata["case_id"])
            results.append(
                GenerationResult(
                    text=text,
                    latency_seconds=0.01,
                    prompt_tokens=len(request.user_prompt.split()),
                    completion_tokens=1,
                )
            )
        return tuple(results)


class _RecordingToolEnvironment(ToolEnvironment):
    def __init__(self):
        self.calls = []

    def execute(self, action, arguments, metadata):
        self.calls.append((action, arguments, metadata["case_id"]))
        return "result-for-{}".format(arguments)


def test_tool_loop_batches_roles_and_grows_one_shared_scratchpad():
    adapters = (
        AdapterConfig(name="a", path="a"),
        AdapterConfig(name="b", path="b"),
    )
    prompt = "Task:\n{task}\n\nShared scratchpad:\n{scratchpad}"
    agents = (
        AgentConfig("planner", "planner", "plan", prompt),
        AgentConfig("arguments", "argument generator", "args", prompt),
        AgentConfig("final", "final answerer", "final", prompt),
    )
    topology = ToolLoopTopology(
        tuple(agent.name for agent in agents),
        TopologyConfig(
            edges=(),
            output_agent="final",
            kind="tool_loop",
            loop_agents=("planner", "arguments"),
            max_steps=2,
            max_context_tokens=4096,
        ),
    )
    backend = _LoopBackend(adapters)
    environment = _RecordingToolEnvironment()
    system = ToolLoopSystem(
        backend,
        agents,
        topology,
        GenerationConfig(seed=4),
        tool_environment=environment,
    )
    plan = FusionPlan(
        tuple(
            (
                agent.name,
                AdapterMixture(
                    ("a", "b"),
                    (1.0, 0.0) if agent.name == "planner" else (0.0, 1.0),
                ),
            )
            for agent in agents
        )
    )

    results = system.run_many(
        ("task one", "task two"),
        plan,
        ({"case_id": "one"}, {"case_id": "two"}),
    )

    assert backend.batch_sizes == [2, 2, 2, 2, 2]
    assert [result.output for result in results] == ["final-one", "final-two"]
    assert all(len(result.traces) == 5 for result in results)
    assert "Step 1 | Plan/Action\nlookup" in results[0].traces[1].prompt
    assert "Step 1 | Observation\nresult-for-query-one" in (
        results[0].traces[2].prompt
    )
    assert "Step 2 | Observation\nresult-for-query-one" in (
        results[0].traces[-1].prompt
    )
    assert len(environment.calls) == 4
    assert all(result.termination_reason == "max_steps" for result in results)
    assert all(result.steps == 2 for result in results)
    assert results[0].traces[0].mixture.as_dict() == {"a": 1.0}
    assert results[0].traces[1].mixture.as_dict() == {"b": 1.0}


class _StoppingLoopBackend(_LoopBackend):
    def __init__(self, adapters):
        super().__init__(adapters)
        self.planner_calls = {}

    def generate_many(self, requests, mixtures):
        self.batch_sizes.append(len(requests))
        results = []
        for request, mixture in zip(requests, mixtures):
            self._pool.validate_policy(mixture)
            case_id = request.metadata["case_id"]
            if request.agent_name == "planner":
                count = self.planner_calls.get(case_id, 0) + 1
                self.planner_calls[case_id] = count
                text = (
                    "Finish[ready]"
                    if case_id == "one" or count == 2
                    else "Reason"
                )
            elif request.agent_name == "arguments":
                text = "answer-{}".format(case_id)
            else:
                text = "verified-{}".format(case_id)
            results.append(
                GenerationResult(text, 0.01, 4, 1)
            )
        return tuple(results)


def test_tool_loop_finish_stops_cases_without_blocking_batch_peers():
    adapters = (AdapterConfig(name="a", path="a"),)
    prompt = "Task: {task}\nShared scratchpad:\n{scratchpad}"
    agents = (
        AgentConfig("planner", "planner", "plan", prompt),
        AgentConfig("arguments", "arguments", "arguments", prompt),
        AgentConfig("final", "final", "final", prompt),
    )
    topology = ToolLoopTopology(
        tuple(agent.name for agent in agents),
        TopologyConfig(
            edges=(),
            output_agent="final",
            loop_agents=("planner", "arguments"),
            max_steps=3,
            max_context_tokens=4096,
        ),
    )
    backend = _StoppingLoopBackend(adapters)
    environment = _RecordingToolEnvironment()
    system = ToolLoopSystem(
        backend, agents, topology, GenerationConfig(), environment
    )
    plan = FusionPlan(
        tuple(
            (agent.name, AdapterMixture(("a",), (1.0,)))
            for agent in agents
        )
    )

    results = system.run_many(
        ("one", "two"),
        plan,
        ({"case_id": "one"}, {"case_id": "two"}),
    )

    assert backend.batch_sizes == [2, 2, 1, 1, 2]
    assert [result.termination_reason for result in results] == [
        "finish",
        "finish",
    ]
    assert [result.steps for result in results] == [1, 2]
    assert [result.output for result in results] == [
        "verified-one",
        "verified-two",
    ]
    assert environment.calls == [("Reason", "answer-two", "two")]


class _NoResponseBackend(_LoopBackend):
    def generate_many(self, requests, mixtures):
        self.batch_sizes.append(len(requests))
        return tuple(
            GenerationResult("No response", 0.01, 4, 0)
            for _ in requests
        )


class _FailingToolEnvironment(ToolEnvironment):
    def execute(self, action, arguments, metadata):
        raise RuntimeError("simulated tool failure")


def test_tool_loop_records_generation_and_context_halts():
    adapters = (AdapterConfig(name="a", path="a"),)
    agents = (
        AgentConfig("planner", "planner", "plan"),
        AgentConfig("arguments", "arguments", "arguments"),
        AgentConfig("final", "final", "final"),
    )
    plan = FusionPlan(
        tuple(
            (agent.name, AdapterMixture(("a",), (1.0,)))
            for agent in agents
        )
    )

    error_backend = _NoResponseBackend(adapters)
    error_system = ToolLoopSystem(
        error_backend,
        agents,
        ToolLoopTopology(
            tuple(agent.name for agent in agents),
            TopologyConfig(
                edges=(),
                output_agent="final",
                loop_agents=("planner", "arguments"),
                max_steps=2,
                max_context_tokens=4096,
            ),
        ),
        GenerationConfig(),
    )
    error_result = error_system.run("task", plan)
    assert error_result.termination_reason == "generation_error"
    assert len(error_result.traces) == 1

    context_backend = _LoopBackend(adapters)
    context_system = ToolLoopSystem(
        context_backend,
        agents,
        ToolLoopTopology(
            tuple(agent.name for agent in agents),
            TopologyConfig(
                edges=(),
                output_agent="final",
                loop_agents=("planner", "arguments"),
                max_steps=2,
                max_context_tokens=1,
            ),
        ),
        GenerationConfig(),
    )
    context_result = context_system.run("task", plan)
    assert context_result.termination_reason == "context_limit"
    assert not context_result.traces
    assert context_backend.batch_sizes == []


def test_tool_failure_becomes_observation_instead_of_aborting_case():
    adapters = (AdapterConfig(name="a", path="a"),)
    agents = (
        AgentConfig("planner", "planner", "plan"),
        AgentConfig("arguments", "arguments", "arguments"),
        AgentConfig("final", "final", "final"),
    )
    topology = ToolLoopTopology(
        tuple(agent.name for agent in agents),
        TopologyConfig(
            edges=(),
            output_agent="final",
            loop_agents=("planner", "arguments"),
            max_steps=1,
            max_context_tokens=4096,
        ),
    )
    backend = _LoopBackend(adapters)
    system = ToolLoopSystem(
        backend,
        agents,
        topology,
        GenerationConfig(),
        _FailingToolEnvironment(),
    )
    plan = FusionPlan(
        tuple(
            (agent.name, AdapterMixture(("a",), (1.0,)))
            for agent in agents
        )
    )

    result = system.run("task", plan, {"case_id": "one"})

    assert result.termination_reason == "max_steps"
    assert result.tool_error_count == 1
    assert "Tool error (RuntimeError)" in result.traces[-1].prompt
