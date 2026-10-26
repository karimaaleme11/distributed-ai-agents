"""
Abstract base layer for autonomous agents in the distributed coordination system.
Async priority task queue, skill registry, and observable task history.
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

logger = logging.getLogger(__name__)


class TaskState(Enum):
    PENDING  = auto()
    RUNNING  = auto()
    COMPLETE = auto()
    FAILED   = auto()


@dataclass
class AgentTask:
    task_id:      str            = field(default_factory=lambda: str(uuid.uuid4()))
    name:         str            = ""
    payload:      dict[str, Any] = field(default_factory=dict)
    priority:     int            = 0
    state:        TaskState      = TaskState.PENDING
    created_at:   datetime       = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    result:       Any            = None
    error:        str | None     = None

    def __lt__(self, other: AgentTask) -> bool:
        return self.priority > other.priority


@dataclass
class SkillDescriptor:
    name:        str
    description: str
    handler:     Callable[..., Coroutine[Any, Any, Any]]


class BaseAgent(ABC):
    """
    Abstract agent with an async priority queue, bounded concurrency semaphore,
    a named-skill registry, and a full task-history log.
    """

    MAX_CONCURRENT_TASKS: int = 4

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "BaseAgent",
        model: str = "gemini-2.5-flash",
    ) -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name     = name
        self.model    = model

        self._task_queue:   asyncio.PriorityQueue[AgentTask] = asyncio.PriorityQueue()
        self._task_history: list[AgentTask]                  = []
        self._skills:       dict[str, SkillDescriptor]       = {}
        self._semaphore:    asyncio.Semaphore                 = asyncio.Semaphore(self.MAX_CONCURRENT_TASKS)
        self._worker_task:  asyncio.Task[None] | None         = None

        self._register_skills()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _register_skills(self) -> None:
        """Populate self._skills during __init__."""

    @abstractmethod
    async def _process_task(self, task: AgentTask) -> Any:
        """Domain-specific task execution."""

    # ------------------------------------------------------------------
    # Skill helpers
    # ------------------------------------------------------------------

    def add_skill(self, descriptor: SkillDescriptor) -> None:
        self._skills[descriptor.name] = descriptor

    def get_skill(self, name: str) -> SkillDescriptor | None:
        return self._skills.get(name)

    # ------------------------------------------------------------------
    # Queue interface
    # ------------------------------------------------------------------

    async def enqueue(self, task: AgentTask) -> None:
        task.state = TaskState.PENDING
        await self._task_queue.put(task)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._worker_task and not self._worker_task.done():
            return
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name=f"worker-{self.agent_id[:8]}"
        )
        logger.info("[%s] started", self.name)

    async def stop(self) -> None:
        await self._task_queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[%s] stopped", self.name)

    async def _worker_loop(self) -> None:
        while True:
            task = await self._task_queue.get()
            asyncio.create_task(self._execute(task))

    async def _execute(self, task: AgentTask) -> None:
        async with self._semaphore:
            task.state = TaskState.RUNNING
            try:
                task.result = await self._process_task(task)
                task.state  = TaskState.COMPLETE
            except Exception as exc:  # noqa: BLE001
                task.state = TaskState.FAILED
                task.error = str(exc)
                logger.exception("[%s] task '%s' failed: %s", self.name, task.name, exc)
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
            "name":     self.name,
            "model":    self.model,
            "pending":  self.pending_count,
            "history":  counts,
            "skills":   list(self._skills.keys()),
        }
