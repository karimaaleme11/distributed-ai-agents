"""
Abstract base layer for autonomous agents in the distributed coordination system.
Extends the google-antigravity SDK Agent class with async task queues and skill registration.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine

from google.antigravity import Agent
from google.antigravity.skills import Skill, SkillSet

logger = logging.getLogger(__name__)


class TaskState(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass
class AgentTask:
    """Represents a discrete unit of work dispatched to an agent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    def __lt__(self, other: AgentTask) -> bool:
        # Higher priority value → dequeued first.
        return self.priority > other.priority


@dataclass
class AgentSkillDescriptor:
    """Metadata wrapper around a registered Skill."""

    skill: Skill
    handler: Callable[..., Coroutine[Any, Any, Any]]
    description: str


class BaseAgent(Agent, ABC):
    """
    Abstract agent that couples the google-antigravity Agent lifecycle with:
      - an async priority task queue (asyncio.PriorityQueue)
      - a SkillSet registry consumed by the Antigravity runtime
      - observable task-history for the simulation visualizer

    Concrete subclasses must implement:
      • `_build_skillset`  – return a populated SkillSet
      • `_process_task`    – domain-specific task handling logic
    """

    MAX_CONCURRENT_TASKS: int = 4

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "BaseAgent",
        model: str = "gemini-2.5-pro",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, model=model, **kwargs)

        self.agent_id: str = agent_id or str(uuid.uuid4())
        self._task_queue: asyncio.PriorityQueue[AgentTask] = asyncio.PriorityQueue()
        self._task_history: list[AgentTask] = []
        self._skill_registry: dict[str, AgentSkillDescriptor] = {}
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_TASKS)
        self._worker_task: asyncio.Task[None] | None = None

        skillset = self._build_skillset()
        self.register_skills(skillset)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_skillset(self) -> SkillSet:
        """Construct and return the SkillSet for this agent persona."""

    @abstractmethod
    async def _process_task(self, task: AgentTask) -> Any:
        """Execute a single task and return its result."""

    # ------------------------------------------------------------------
    # Skill management
    # ------------------------------------------------------------------

    def register_skill(self, descriptor: AgentSkillDescriptor) -> None:
        """Attach a skill to the agent and make it callable by the runtime."""
        self._skill_registry[descriptor.skill.name] = descriptor
        logger.debug("[%s] registered skill '%s'", self.agent_id, descriptor.skill.name)

    def get_skill(self, name: str) -> AgentSkillDescriptor | None:
        return self._skill_registry.get(name)

    # ------------------------------------------------------------------
    # Task queue interface
    # ------------------------------------------------------------------

    async def enqueue(self, task: AgentTask) -> None:
        """Push a task onto the priority queue."""
        task.state = TaskState.PENDING
        await self._task_queue.put(task)
        logger.debug("[%s] enqueued task '%s' (priority=%d)", self.agent_id, task.name, task.priority)

    async def enqueue_many(self, tasks: list[AgentTask]) -> None:
        for task in tasks:
            await self.enqueue(task)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch the background worker loop."""
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(self._worker_loop(), name=f"worker-{self.agent_id}")
        logger.info("[%s] started", self.agent_id)

    async def stop(self) -> None:
        """Drain the queue then cancel the worker."""
        await self._task_queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[%s] stopped", self.agent_id)

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        while True:
            task = await self._task_queue.get()
            asyncio.create_task(self._execute(task))

    async def _execute(self, task: AgentTask) -> None:
        async with self._semaphore:
            task.state = TaskState.RUNNING
            logger.debug("[%s] running task '%s'", self.agent_id, task.name)
            try:
                task.result = await self._process_task(task)
                task.state = TaskState.COMPLETE
            except Exception as exc:  # noqa: BLE001
                task.state = TaskState.FAILED
                task.error = str(exc)
                logger.exception("[%s] task '%s' failed: %s", self.agent_id, task.name, exc)
            finally:
                task.completed_at = datetime.utcnow()
                self._task_history.append(task)
                self._task_queue.task_done()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def task_history(self) -> list[AgentTask]:
        return list(self._task_history)

    @property
    def pending_count(self) -> int:
        return self._task_queue.qsize()

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {s.name: 0 for s in TaskState}
        for t in self._task_history:
            counts[t.state.name] += 1
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model": self.model,
            "pending": self.pending_count,
            "history": counts,
            "skills": list(self._skill_registry.keys()),
        }
