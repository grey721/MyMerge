"""DAG and shared-scratchpad multi-agent execution systems."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ..backends.base import ModelBackend
from ..core.config import (
    AgentConfig,
    ExperimentConfig,
    GenerationConfig,
    TopologyConfig,
)
from ..core.domain import (
    AgentTrace,
    GenerationRequest,
    GenerationResult,
    SystemRunResult,
)
from ..core.policies import FusionPlan, FusionPolicy


@dataclass(frozen=True)
class AgentNode:
    config: AgentConfig
    parents: Tuple[str, ...]

    def render_prompt(
        self,
        task: str,
        parent_outputs: Mapping[str, str],
        scratchpad: Optional[str] = None,
        step_index: Optional[int] = None,
        incoming_text: Optional[str] = None,
    ) -> str:
        incoming = incoming_text
        if incoming is None:
            incoming = "\n\n".join(
                "[{}]\n{}".format(parent, parent_outputs[parent])
                for parent in self.parents
            )
        if not incoming:
            incoming = "(none)"
        shared_context = scratchpad if scratchpad else "(empty)"
        return (
            self.config.prompt_template.replace("{task}", task)
            .replace("{incoming}", incoming)
            .replace("{role}", self.config.role)
            .replace("{scratchpad}", shared_context)
            .replace(
                "{step}",
                "" if step_index is None else str(step_index),
            )
        )


class FixedTopology:
    """Validated directed acyclic graph with deterministic execution order."""

    def __init__(
        self, agent_names: Sequence[str], config: TopologyConfig
    ) -> None:
        self.agent_names = tuple(agent_names)
        self.output_agent = config.output_agent
        self.edges = tuple(config.edges)
        self._parents = {name: [] for name in self.agent_names}
        self._children = {name: [] for name in self.agent_names}
        for source, target in self.edges:
            self._parents[target].append(source)
            self._children[source].append(target)
        self._order = self._topological_order()

    @property
    def execution_order(self) -> Tuple[str, ...]:
        return self._order

    def parents_of(self, agent_name: str) -> Tuple[str, ...]:
        return tuple(self._parents[agent_name])

    def _topological_order(self) -> Tuple[str, ...]:
        indegree = {
            name: len(self._parents[name]) for name in self.agent_names
        }
        queue = [name for name in self.agent_names if indegree[name] == 0]
        order: List[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in self._children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.agent_names):
            raise ValueError("Agent topology must be acyclic")
        return tuple(order)


class MultiAgentSystem:
    """Execute one immutable topology with role-specific adapter mixtures."""

    def __init__(
        self,
        backend: ModelBackend,
        agents: Sequence[AgentConfig],
        topology: FixedTopology,
        generation: GenerationConfig,
    ) -> None:
        self.backend = backend
        self.agents = {agent.name: agent for agent in agents}
        self.topology = topology
        self.generation = generation
        self._nodes = {
            name: AgentNode(
                config=self.agents[name],
                parents=topology.parents_of(name),
            )
            for name in topology.agent_names
        }

    @contextmanager
    def candidate_session(self):
        """Expose the backend's candidate lifecycle at the system boundary."""

        with self.backend.candidate_session():
            yield

    def run(
        self,
        task: str,
        plan: FusionPlan,
        metadata: Mapping[str, object] = None,
    ) -> SystemRunResult:
        """Execute one task through the stage-batched topology."""

        return self.run_many((task,), plan, (metadata or {},))[0]

    def run_many(
        self,
        tasks: Sequence[str],
        plan: FusionPlan,
        metadata: Sequence[Mapping[str, object]] = None,
    ) -> Tuple[SystemRunResult, ...]:
        """Execute tasks in topology-stage batches.

        All tasks finish the current agent before the next agent runs. This
        preserves DAG dependencies while allowing the backend to batch the
        independent prompts at each stage.
        """

        task_batch = tuple(tasks)
        if not task_batch:
            return ()
        if metadata is None:
            metadata_batch = tuple({} for _ in task_batch)
        else:
            if len(metadata) != len(task_batch):
                raise ValueError("metadata and tasks must have the same length")
            metadata_batch = tuple(dict(item) for item in metadata)

        outputs = [{} for _ in task_batch]  # type: List[Dict[str, str]]
        traces = [[] for _ in task_batch]  # type: List[List[AgentTrace]]
        for index, agent_name in enumerate(self.topology.execution_order):
            node = self._nodes[agent_name]
            policy = self.policy_for(plan, agent_name)
            prompts = tuple(
                node.render_prompt(task, case_outputs)
                for task, case_outputs in zip(task_batch, outputs)
            )
            requests = tuple(
                GenerationRequest(
                    agent_name=agent_name,
                    system_prompt=node.config.system_prompt,
                    user_prompt=prompt,
                    max_new_tokens=(
                        node.config.max_new_tokens
                        if node.config.max_new_tokens is not None
                        else self.generation.max_new_tokens
                    ),
                    temperature=self.generation.temperature,
                    seed=(
                        None
                        if self.generation.seed is None
                        else self.generation.seed + index
                    ),
                    metadata=request_metadata,
                )
                for prompt, request_metadata in zip(prompts, metadata_batch)
            )
            generations = self.backend.generate_many(
                requests, tuple(policy for _ in requests)
            )
            if len(generations) != len(task_batch):
                raise RuntimeError(
                    "Backend returned {} generations for {} requests".format(
                        len(generations), len(task_batch)
                    )
                )
            for case_index, (prompt, generation) in enumerate(
                zip(prompts, generations)
            ):
                outputs[case_index][agent_name] = generation.text
                traces[case_index].append(
                    AgentTrace(
                        agent_name=agent_name,
                        prompt=prompt,
                        mixture=policy,
                        generation=generation,
                        parent_agents=node.parents,
                    )
                )

        return tuple(
            SystemRunResult(
                output=case_outputs[self.topology.output_agent],
                traces=tuple(case_traces),
                termination_reason="completed",
                steps=len(self.topology.execution_order),
            )
            for case_outputs, case_traces in zip(outputs, traces)
        )

    def policy_for(
        self, plan: FusionPlan, agent_name: str
    ) -> FusionPolicy:
        """Select one Agent policy from an optimizer-independent plan."""

        policy = plan.policy_for(agent_name)
        if tuple(policy.adapter_names) != tuple(self.backend.adapter_names):
            raise ValueError("FusionPlan and backend Adapter pools differ")
        return policy


class ToolEnvironment(ABC):
    """Execute a planner-selected action outside the language model."""

    @abstractmethod
    def execute(
        self,
        action: str,
        arguments: str,
        metadata: Mapping[str, object],
    ) -> str:
        """Return the observation appended to the shared scratchpad."""


class MetadataToolEnvironment(ToolEnvironment):
    """Replay tool observations supplied by a benchmark case.

    ``metadata["tool_results"]`` may map an action directly to a result, or
    map it to a second dictionary keyed by the exact argument text.  This
    neutral default keeps the runtime independent of BFCL and lets projects
    inject a real environment without changing the optimizer or backend.
    """

    def execute(
        self,
        action: str,
        arguments: str,
        metadata: Mapping[str, object],
    ) -> str:
        action_name = _normalize_action(action)
        if action_name.lower() in ("reason", "think"):
            return arguments.strip()
        results = metadata.get("tool_results", {})
        if isinstance(results, Mapping):
            result = results.get(action_name, results.get(action.strip()))
            if isinstance(result, Mapping):
                result = result.get(arguments.strip(), result.get("default"))
            if result is not None:
                return str(result)
        return "No tool result is available for action {!r}.".format(
            action_name
        )


class ToolLoopTopology:
    """Planner/argument-generator loop followed by one output Agent."""

    def __init__(
        self, agent_names: Sequence[str], config: TopologyConfig
    ) -> None:
        self.agent_names = tuple(agent_names)
        self.output_agent = config.output_agent
        if len(config.loop_agents) != 2:
            raise ValueError(
                "ToolLoopTopology requires planner and argument generator"
            )
        self.planner_agent, self.argument_agent = config.loop_agents
        self.max_steps = int(config.max_steps)
        self.max_context_tokens = int(config.max_context_tokens)
        if self.max_steps < 1:
            raise ValueError("ToolLoopTopology max_steps must be positive")
        if self.max_context_tokens < 1:
            raise ValueError(
                "ToolLoopTopology max_context_tokens must be positive"
            )
        used = {
            self.planner_agent,
            self.argument_agent,
            self.output_agent,
        }
        if used != set(self.agent_names):
            raise ValueError(
                "ToolLoopTopology must use every configured agent exactly as "
                "planner, argument generator, or output Agent"
            )


@dataclass
class _LoopCaseState:
    task: str
    metadata: Mapping[str, object]
    scratchpad: List[str]
    traces: List[AgentTrace]
    active: bool = True
    termination_reason: str = "running"
    steps: int = 0
    tool_error_count: int = 0
    finish_answer: str = ""
    last_output: str = ""
    final_output: str = ""


class ToolLoopSystem:
    """Run a batched shared-scratchpad tool loop.

    Every round first asks the planner for an action, then asks a distinct
    Agent for its arguments.  The external observation is appended before the
    next round.  Candidate adapter materialization remains owned by the same
    backend candidate session as the fixed DAG system.
    """

    def __init__(
        self,
        backend: ModelBackend,
        agents: Sequence[AgentConfig],
        topology: ToolLoopTopology,
        generation: GenerationConfig,
        tool_environment: Optional[ToolEnvironment] = None,
    ) -> None:
        self.backend = backend
        self.agents = {agent.name: agent for agent in agents}
        self.topology = topology
        self.generation = generation
        self.tool_environment = tool_environment or MetadataToolEnvironment()
        self._nodes = {
            name: AgentNode(self.agents[name], ())
            for name in topology.agent_names
        }

    @contextmanager
    def candidate_session(self):
        with self.backend.candidate_session():
            yield

    def run(
        self,
        task: str,
        plan: FusionPlan,
        metadata: Mapping[str, object] = None,
    ) -> SystemRunResult:
        return self.run_many((task,), plan, (metadata or {},))[0]

    def run_many(
        self,
        tasks: Sequence[str],
        plan: FusionPlan,
        metadata: Sequence[Mapping[str, object]] = None,
    ) -> Tuple[SystemRunResult, ...]:
        task_batch = tuple(tasks)
        if not task_batch:
            return ()
        if metadata is None:
            metadata_batch = tuple({} for _ in task_batch)
        else:
            if len(metadata) != len(task_batch):
                raise ValueError("metadata and tasks must have the same length")
            metadata_batch = tuple(dict(item) for item in metadata)

        states = [
            _LoopCaseState(task, request_metadata, [], [])
            for task, request_metadata in zip(task_batch, metadata_batch)
        ]
        generation_index = 0
        for step_index in range(1, self.topology.max_steps + 1):
            active_indices = tuple(
                index for index, state in enumerate(states) if state.active
            )
            if not active_indices:
                break
            planner_results, context_limited = self._generate_phase(
                states,
                active_indices,
                plan,
                self.topology.planner_agent,
                step_index,
                generation_index,
                (),
            )
            generation_index += 1
            self._mark_context_limited(states, context_limited)
            planner_actions = {}  # type: Dict[int, str]
            for case_index, result in planner_results.items():
                state = states[case_index]
                state.last_output = result.text.strip()
                if _is_generation_error(result.text):
                    self._mark_generation_error(state, step_index, "planner")
                    continue
                action = _normalize_action(result.text)
                planner_actions[case_index] = action
                state.scratchpad.append(
                    "Step {} | Plan/Action\n{}".format(step_index, action)
                )

            argument_indices = tuple(
                index
                for index in active_indices
                if states[index].active and index in planner_actions
            )
            argument_results, context_limited = self._generate_phase(
                states,
                argument_indices,
                plan,
                self.topology.argument_agent,
                step_index,
                generation_index,
                (self.topology.planner_agent,),
            )
            generation_index += 1
            self._mark_context_limited(states, context_limited)
            for case_index, result in argument_results.items():
                state = states[case_index]
                state.last_output = result.text.strip()
                state.steps = step_index
                if _is_generation_error(result.text):
                    self._mark_generation_error(
                        state, step_index, "argument generator"
                    )
                    continue
                arguments = result.text.strip()
                state.scratchpad.append(
                    "Step {} | Arguments\n{}".format(step_index, arguments)
                )
                action = planner_actions[case_index]
                if action.lower() == "finish":
                    state.finish_answer = arguments
                    state.active = False
                    state.termination_reason = "finish"
                    state.scratchpad.append(
                        "Step {} | Observation\nFinish requested.".format(
                            step_index
                        )
                    )
                    continue
                try:
                    observation = self.tool_environment.execute(
                        action, arguments, state.metadata
                    )
                except Exception as exc:
                    state.tool_error_count += 1
                    observation = (
                        "Tool error ({}); revise the next action.".format(
                            type(exc).__name__
                        )
                    )
                state.scratchpad.append(
                    "Step {} | Observation\n{}".format(
                        step_index, observation
                    )
                )

        for state in states:
            if state.active:
                state.active = False
                state.termination_reason = "max_steps"

        final_indices = tuple(
            index
            for index, state in enumerate(states)
            if state.termination_reason in ("finish", "max_steps")
        )
        final_results, context_limited = self._generate_phase(
            states,
            final_indices,
            plan,
            self.topology.output_agent,
            self.topology.max_steps + 1,
            generation_index,
            (self.topology.argument_agent,),
        )
        self._mark_context_limited(states, context_limited)
        for case_index, result in final_results.items():
            state = states[case_index]
            if _is_generation_error(result.text):
                state.termination_reason = "generation_error"
            else:
                state.final_output = result.text.strip()

        return tuple(
            SystemRunResult(
                output=(
                    state.final_output
                    or state.finish_answer
                    or state.last_output
                ),
                traces=tuple(state.traces),
                termination_reason=state.termination_reason,
                steps=state.steps,
                tool_error_count=state.tool_error_count,
            )
            for state in states
        )

    def _generate_phase(
        self,
        states: Sequence[_LoopCaseState],
        case_indices: Sequence[int],
        plan: FusionPlan,
        agent_name: str,
        step_index: int,
        generation_index: int,
        parent_agents: Tuple[str, ...],
    ) -> Tuple[Dict[int, GenerationResult], Tuple[int, ...]]:
        if not case_indices:
            return {}, ()
        node = self._nodes[agent_name]
        policy = self.policy_for(plan, agent_name)
        eligible_indices = []
        prompts = []
        requests = []
        context_limited = []
        for case_index in case_indices:
            state = states[case_index]
            rendered_scratchpad = _render_scratchpad(state.scratchpad)
            prompt = node.render_prompt(
                state.task,
                {},
                scratchpad=rendered_scratchpad,
                step_index=step_index,
                incoming_text=rendered_scratchpad,
            )
            request = GenerationRequest(
                agent_name=agent_name,
                system_prompt=node.config.system_prompt,
                user_prompt=prompt,
                max_new_tokens=(
                    node.config.max_new_tokens
                    if node.config.max_new_tokens is not None
                    else self.generation.max_new_tokens
                ),
                temperature=self.generation.temperature,
                seed=(
                    None
                    if self.generation.seed is None
                    else self.generation.seed + generation_index
                ),
                metadata=state.metadata,
            )
            if (
                self.backend.count_prompt_tokens(request)
                > self.topology.max_context_tokens
            ):
                context_limited.append(case_index)
                continue
            eligible_indices.append(case_index)
            prompts.append(prompt)
            requests.append(request)
        if not requests:
            return {}, tuple(context_limited)
        generations = self.backend.generate_many(
            tuple(requests), tuple(policy for _ in requests)
        )
        if len(generations) != len(requests):
            raise RuntimeError(
                "Backend returned {} generations for {} requests".format(
                    len(generations), len(requests)
                )
            )
        results = {}  # type: Dict[int, GenerationResult]
        for case_index, prompt, generation in zip(
            eligible_indices, prompts, generations
        ):
            states[case_index].traces.append(
                AgentTrace(
                    agent_name=agent_name,
                    prompt=prompt,
                    mixture=policy,
                    generation=generation,
                    parent_agents=parent_agents,
                )
            )
            results[case_index] = generation
        return results, tuple(context_limited)

    @staticmethod
    def _mark_context_limited(
        states: Sequence[_LoopCaseState], case_indices: Sequence[int]
    ) -> None:
        for case_index in case_indices:
            state = states[case_index]
            state.active = False
            state.termination_reason = "context_limit"

    @staticmethod
    def _mark_generation_error(
        state: _LoopCaseState, step_index: int, phase: str
    ) -> None:
        state.active = False
        state.termination_reason = "generation_error"
        state.scratchpad.append(
            "Step {} | Error\nNo response from {}.".format(
                step_index, phase
            )
        )

    def policy_for(
        self, plan: FusionPlan, agent_name: str
    ) -> FusionPolicy:
        policy = plan.policy_for(agent_name)
        if tuple(policy.adapter_names) != tuple(self.backend.adapter_names):
            raise ValueError("FusionPlan and backend Adapter pools differ")
        return policy


AgentSystem = Union[MultiAgentSystem, ToolLoopSystem]


def create_agent_system(
    config: ExperimentConfig,
    backend: ModelBackend,
    tool_environment: Optional[ToolEnvironment] = None,
) -> AgentSystem:
    """Construct the YAML-selected system without coupling callers to it."""

    kind = config.topology.kind.lower().replace("-", "_")
    if kind == "fixed":
        topology = FixedTopology(config.agent_names, config.topology)
        return MultiAgentSystem(
            backend, config.agents, topology, config.model.generation
        )
    if kind == "tool_loop":
        topology = ToolLoopTopology(config.agent_names, config.topology)
        return ToolLoopSystem(
            backend,
            config.agents,
            topology,
            config.model.generation,
            tool_environment=tool_environment,
        )
    raise ValueError("Unknown topology kind {!r}".format(config.topology.kind))


def _render_scratchpad(entries: Sequence[str]) -> str:
    return "\n\n".join(entries) if entries else "(empty)"


def _normalize_action(value: str) -> str:
    """Accept LRAgent-style ``Finish[x]`` and bare action names."""

    action = str(value).strip().splitlines()[0] if str(value).strip() else ""
    if ":" in action and action.split(":", 1)[0].strip().lower() in (
        "action",
        "plan/action",
    ):
        action = action.split(":", 1)[1].strip()
    if "[" in action:
        action = action.split("[", 1)[0].strip()
    return action.rstrip(" .:")


def _is_generation_error(value: str) -> bool:
    normalized = str(value).strip().lower()
    return not normalized or normalized == "no response"
