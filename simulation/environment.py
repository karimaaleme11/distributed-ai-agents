"""
Shared workspace environment where agents collaborate on code/data workflows.
Provides a structured artifact store, event log, and agent channel primitives.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactKind(Enum):
    CODE = auto()
    DATA = auto()
    REPORT = auto()
    LOG = auto()


@dataclass
class Artifact:
    """Immutable versioned unit of work produced by an agent."""

    artifact_id: str
    kind: ArtifactKind
    owner_agent_id: str
    content: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        raw = str(self.content).encode()
        return hashlib.sha256(raw).hexdigest()[:12]


@dataclass
class EnvironmentEvent:
    """Audit record for state transitions within the workspace."""

    event_id: str
    event_type: str
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkspaceEnvironment:
    """
    Shared environment providing:
      - A versioned artifact store keyed by artifact_id
      - An ordered event log for replay / visualization
      - Per-agent async channels for direct peer messaging
      - A global broadcast bus for workspace-wide notifications

    Agents interact via `put_artifact`, `get_artifact`, `send_to`, and `subscribe`.
    """

    def __init__(self, workspace_id: str = "default") -> None:
        self.workspace_id = workspace_id
        self._artifacts: dict[str, list[Artifact]] = {}   # artifact_id → version list
        self._event_log: list[EnvironmentEvent] = []
        self._channels: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._broadcast_bus: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}

    # ------------------------------------------------------------------
    # Agent channel management
    # ------------------------------------------------------------------

    def register_agent_channel(self, agent_id: str) -> None:
        if agent_id not in self._channels:
            self._channels[agent_id] = asyncio.Queue()
            logger.debug("[env:%s] channel created for agent %s", self.workspace_id, agent_id)

    async def send_to(self, target_agent_id: str, message: dict[str, Any]) -> None:
        channel = self._channels.get(target_agent_id)
        if channel is None:
            raise KeyError(f"No channel for agent '{target_agent_id}'")
        await channel.put(message)
        self._log_event("message_sent", sender_id=message.get("from", "env"), target=target_agent_id)

    async def receive(self, agent_id: str) -> dict[str, Any]:
        channel = self._channels.get(agent_id)
        if channel is None:
            raise KeyError(f"No channel for agent '{agent_id}'")
        return await channel.get()

    # ------------------------------------------------------------------
    # Broadcast pub/sub
    # ------------------------------------------------------------------

    def subscribe(self, agent_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.setdefault(agent_id, []).append(q)
        return q

    async def broadcast(self, message: dict[str, Any]) -> None:
        await self._broadcast_bus.put(message)
        for queues in self._subscribers.values():
            for q in queues:
                await q.put(message)

    # ------------------------------------------------------------------
    # Artifact store
    # ------------------------------------------------------------------

    def put_artifact(self, artifact: Artifact) -> None:
        self._artifacts.setdefault(artifact.artifact_id, []).append(artifact)
        self._log_event(
            "artifact_stored",
            sender_id=artifact.owner_agent_id,
            artifact_id=artifact.artifact_id,
            kind=artifact.kind.name,
            version=len(self._artifacts[artifact.artifact_id]),
        )
        logger.debug(
            "[env:%s] artifact '%s' v%d stored by agent %s",
            self.workspace_id,
            artifact.artifact_id,
            len(self._artifacts[artifact.artifact_id]),
            artifact.owner_agent_id,
        )

    def get_artifact(self, artifact_id: str, version: int = -1) -> Artifact | None:
        versions = self._artifacts.get(artifact_id)
        if not versions:
            return None
        return versions[version]

    def list_artifacts(self, kind: ArtifactKind | None = None) -> list[Artifact]:
        all_latest = [vs[-1] for vs in self._artifacts.values()]
        if kind is not None:
            return [a for a in all_latest if a.kind == kind]
        return all_latest

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------

    def _log_event(self, event_type: str, sender_id: str = "env", **metadata: Any) -> None:
        import uuid
        event = EnvironmentEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            agent_id=sender_id,
            metadata=metadata,
        )
        self._event_log.append(event)

    @property
    def event_log(self) -> list[EnvironmentEvent]:
        return list(self._event_log)

    # ------------------------------------------------------------------
    # Workflow helpers
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        steps: list[dict[str, Any]],
        agent_map: dict[str, Any],
    ) -> list[Any]:
        """
        Execute an ordered sequence of workflow steps.

        Each step dict: {"task_name": str, "agent_role": str, "payload": dict}
        agent_map: role → BaseAgent instance
        """
        results: list[Any] = []
        for step in steps:
            role = step["agent_role"]
            agent = agent_map.get(role)
            if agent is None:
                raise ValueError(f"No agent for role '{role}'")

            from .base_agent import AgentTask  # local import to avoid circular
            task = AgentTask(name=step["task_name"], payload=step.get("payload", {}))
            await agent.enqueue(task)
            # Allow the worker loop one cycle.
            await asyncio.sleep(0)
            results.append({"step": step["task_name"], "agent": role, "task_id": task.task_id})

        return results

    def snapshot(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "artifact_count": sum(len(v) for v in self._artifacts.values()),
            "event_count": len(self._event_log),
            "registered_channels": list(self._channels.keys()),
        }
