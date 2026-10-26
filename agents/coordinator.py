"""
Distributed coordination layer built on top of Antigravity subagent delegation.
Routes tasks to the agent team via priority-aware triggers and tracks inter-agent message flow.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from google.antigravity import SubagentHandle
from google.antigravity.triggers import EventTrigger, TriggerCondition

from .base_agent import AgentTask, BaseAgent, TaskState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """Maps a task name pattern to a target agent and optional priority override."""

    pattern: str                          # substring or exact task name
    target_agent_id: str
    priority_override: int | None = None
    transform: Callable[[AgentTask], AgentTask] | None = None


@dataclass
class DelegationRecord:
    """Audit trail entry for a single subagent delegation event."""

    delegated_at: datetime
    source_agent_id: str
    target_agent_id: str
    task_id: str
    task_name: str
    state: TaskState = TaskState.PENDING
    resolved_at: datetime | None = None


# ---------------------------------------------------------------------------
# AgentCoordinator
# ---------------------------------------------------------------------------

class AgentCoordinator:
    """
    Orchestrates a pool of BaseAgent instances using:
      - SubagentHandle delegation for cross-process or remote agents
      - EventTrigger callbacks to react to task completion events
      - RoutingRule evaluation to direct incoming tasks to the correct agent
      - A shared message bus (asyncio.Queue) for inter-agent notifications

    Usage pattern:
        coordinator = AgentCoordinator()
        coordinator.register_agent(analyst, role="log_analyst")
        coordinator.register_agent(decision_agent, role="coordinator")
        coordinator.add_routing_rule(RoutingRule(pattern="parse_logs", target_agent_id=analyst.agent_id))
        await coordinator.start()
        await coordinator.dispatch(AgentTask(name="parse_logs", payload={...}))
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._roles: dict[str, str] = {}               # agent_id → role
        self._role_index: dict[str, str] = {}          # role → agent_id
        self._routing_rules: list[RoutingRule] = []
        self._delegation_log: list[DelegationRecord] = []
        self._message_bus: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._triggers: list[EventTrigger] = []
        self._bus_worker: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent, role: str) -> None:
        self._agents[agent.agent_id] = agent
        self._roles[agent.agent_id] = role
        self._role_index[role] = agent.agent_id
        logger.info("[coordinator] registered agent %s as '%s'", agent.agent_id, role)

    def add_routing_rule(self, rule: RoutingRule) -> None:
        self._routing_rules.append(rule)
        logger.debug("[coordinator] added routing rule: pattern='%s' → %s", rule.pattern, rule.target_agent_id)

    # ------------------------------------------------------------------
    # Trigger configuration
    # ------------------------------------------------------------------

    def register_trigger(
        self,
        condition: TriggerCondition,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        trigger = EventTrigger(condition=condition, callback=handler)
        self._triggers.append(trigger)
        logger.debug("[coordinator] registered trigger on condition '%s'", condition)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        for agent in self._agents.values():
            await agent.start()

        self._bus_worker = asyncio.create_task(
            self._process_message_bus(), name="coordinator-bus"
        )
        logger.info("[coordinator] started with %d agents", len(self._agents))

    async def stop(self) -> None:
        for agent in self._agents.values():
            await agent.stop()

        if self._bus_worker:
            self._bus_worker.cancel()
            try:
                await self._bus_worker
            except asyncio.CancelledError:
                pass
        logger.info("[coordinator] stopped")

    # ------------------------------------------------------------------
    # Dispatch & delegation
    # ------------------------------------------------------------------

    async def dispatch(self, task: AgentTask) -> str:
        """
        Route a task to the appropriate agent.
        Returns the target agent_id.
        """
        target_id = self._resolve_target(task)
        if target_id is None:
            raise ValueError(f"No routing rule matched task '{task.name}'")

        rule = self._find_rule(task)
        if rule and rule.priority_override is not None:
            task.priority = rule.priority_override
        if rule and rule.transform:
            task = rule.transform(task)

        target_agent = self._agents[target_id]

        # Emit delegation via SubagentHandle when agent is remote-capable.
        handle: SubagentHandle | None = getattr(target_agent, "subagent_handle", None)
        if handle is not None:
            await self._delegate_via_handle(handle, task, source_id="coordinator", target_id=target_id)
        else:
            await target_agent.enqueue(task)

        self._delegation_log.append(
            DelegationRecord(
                delegated_at=datetime.utcnow(),
                source_agent_id="coordinator",
                target_agent_id=target_id,
                task_id=task.task_id,
                task_name=task.name,
            )
        )

        await self._message_bus.put({
            "event": "task_dispatched",
            "task_id": task.task_id,
            "task_name": task.name,
            "target_agent_id": target_id,
        })

        return target_id

    async def broadcast(self, task: AgentTask) -> list[str]:
        """Fan-out a task to every registered agent simultaneously."""
        results = await asyncio.gather(
            *[self.dispatch(AgentTask(
                name=task.name,
                payload=task.payload,
                priority=task.priority,
            )) for _ in self._agents],
            return_exceptions=True,
        )
        return [r for r in results if isinstance(r, str)]

    # ------------------------------------------------------------------
    # Subagent delegation
    # ------------------------------------------------------------------

    async def _delegate_via_handle(
        self,
        handle: SubagentHandle,
        task: AgentTask,
        source_id: str,
        target_id: str,
    ) -> None:
        """Invoke the Antigravity SubagentHandle delegation API."""
        await handle.delegate(
            task_id=task.task_id,
            payload=task.payload,
            priority=task.priority,
        )
        logger.debug(
            "[coordinator] delegated task '%s' via SubagentHandle to %s",
            task.name, target_id,
        )

    # ------------------------------------------------------------------
    # Routing resolution
    # ------------------------------------------------------------------

    def _resolve_target(self, task: AgentTask) -> str | None:
        rule = self._find_rule(task)
        if rule:
            return rule.target_agent_id
        # Fallback: route to the agent registered under role 'coordinator'.
        return self._role_index.get("coordinator")

    def _find_rule(self, task: AgentTask) -> RoutingRule | None:
        for rule in self._routing_rules:
            if rule.pattern in task.name:
                return rule
        return None

    # ------------------------------------------------------------------
    # Message bus
    # ------------------------------------------------------------------

    async def _process_message_bus(self) -> None:
        while True:
            message = await self._message_bus.get()
            await self._evaluate_triggers(message)
            self._message_bus.task_done()

    async def _evaluate_triggers(self, message: dict[str, Any]) -> None:
        for trigger in self._triggers:
            if trigger.condition.matches(message):
                try:
                    await trigger.callback(message)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("[coordinator] trigger callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def delegation_log(self) -> list[DelegationRecord]:
        return list(self._delegation_log)

    def agent_summary(self) -> list[dict[str, Any]]:
        return [
            {**agent.summary(), "role": self._roles.get(agent.agent_id)}
            for agent in self._agents.values()
        ]

    def edge_list(self) -> list[tuple[str, str]]:
        """Return (source, target) edges from the delegation log — for graph visualization."""
        return [(r.source_agent_id, r.target_agent_id) for r in self._delegation_log]
