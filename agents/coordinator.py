"""
Distributed coordination layer: priority routing, delegation, and event triggers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

from .base_agent import AgentTask, BaseAgent, TaskState

logger = logging.getLogger(__name__)


@dataclass
class RoutingRule:
    pattern:           str
    target_agent_id:   str
    priority_override: int | None                                        = None
    transform:         Callable[[AgentTask], AgentTask] | None           = None


@dataclass
class TriggerCondition:
    field: str
    value: Any

    def matches(self, message: dict[str, Any]) -> bool:
        return message.get(self.field) == self.value


@dataclass
class EventTrigger:
    condition: TriggerCondition
    callback:  Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


@dataclass
class DelegationRecord:
    delegated_at:    datetime
    source_agent_id: str
    target_agent_id: str
    task_id:         str
    task_name:       str
    state:           TaskState = TaskState.PENDING
    resolved_at:     datetime | None = None


class AgentCoordinator:
    """
    Routes tasks to registered agents via RoutingRules, fires EventTriggers on
    the async message bus, and maintains a full DelegationRecord audit trail.
    """

    def __init__(self) -> None:
        self._agents:          dict[str, BaseAgent]              = {}
        self._roles:           dict[str, str]                    = {}   # agent_id → role
        self._role_index:      dict[str, str]                    = {}   # role → agent_id
        self._routing_rules:   list[RoutingRule]                 = []
        self._delegation_log:  list[DelegationRecord]            = []
        self._message_bus:     asyncio.Queue[dict[str, Any]]     = asyncio.Queue()
        self._triggers:        list[EventTrigger]                = []
        self._bus_worker:      asyncio.Task[None] | None         = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: BaseAgent, role: str) -> None:
        self._agents[agent.agent_id]   = agent
        self._roles[agent.agent_id]    = role
        self._role_index[role]         = agent.agent_id
        logger.info("[coordinator] registered '%s' as '%s'", agent.name, role)

    def add_routing_rule(self, rule: RoutingRule) -> None:
        self._routing_rules.append(rule)

    def register_trigger(self, condition: TriggerCondition, handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]]) -> None:
        self._triggers.append(EventTrigger(condition=condition, callback=handler))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        for agent in self._agents.values():
            await agent.start()
        self._bus_worker = asyncio.create_task(self._process_bus(), name="coordinator-bus")
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

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(self, task: AgentTask) -> str:
        target_id = self._resolve_target(task)
        if target_id is None:
            raise ValueError(f"No routing rule matched task '{task.name}'")

        rule = self._find_rule(task)
        if rule:
            if rule.priority_override is not None:
                task.priority = rule.priority_override
            if rule.transform:
                task = rule.transform(task)

        await self._agents[target_id].enqueue(task)

        self._delegation_log.append(DelegationRecord(
            delegated_at=datetime.utcnow(),
            source_agent_id="coordinator",
            target_agent_id=target_id,
            task_id=task.task_id,
            task_name=task.name,
        ))
        await self._message_bus.put({
            "event":           "task_dispatched",
            "task_id":         task.task_id,
            "task_name":       task.name,
            "target_agent_id": target_id,
        })
        return target_id

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _resolve_target(self, task: AgentTask) -> str | None:
        rule = self._find_rule(task)
        if rule:
            return rule.target_agent_id
        return self._role_index.get("coordinator")

    def _find_rule(self, task: AgentTask) -> RoutingRule | None:
        for rule in self._routing_rules:
            if rule.pattern in task.name:
                return rule
        return None

    # ------------------------------------------------------------------
    # Message bus
    # ------------------------------------------------------------------

    async def _process_bus(self) -> None:
        while True:
            message = await self._message_bus.get()
            for trigger in self._triggers:
                if trigger.condition.matches(message):
                    try:
                        await trigger.callback(message)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("[coordinator] trigger error: %s", exc)
            self._message_bus.task_done()

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
        return [(r.source_agent_id, r.target_agent_id) for r in self._delegation_log]
