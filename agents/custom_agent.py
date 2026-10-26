"""
Concrete agent implementations with distinct operational personas.
Each persona wraps a specific Gemini model configuration and exposes a curated SkillSet.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from google.antigravity.skills import Skill, SkillParameter, SkillSet
from google.antigravity.types import GenerationConfig

from .base_agent import AgentSkillDescriptor, AgentTask, BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared skill factory helpers
# ---------------------------------------------------------------------------

def _make_skill(name: str, description: str, parameters: list[SkillParameter]) -> Skill:
    return Skill(name=name, description=description, parameters=parameters)


# ---------------------------------------------------------------------------
# LogAnalystAgent
# ---------------------------------------------------------------------------

class LogAnalystAgent(BaseAgent):
    """
    Persona: parses raw log streams, extracts anomaly signatures, and emits
    structured diagnostic payloads consumed by the coordinator.

    Model: gemini-2.5-flash  (optimized for high-throughput text ingestion)
    """

    PERSONA_PROMPT = (
        "You are a precision log-analysis engine. "
        "Given raw log lines, extract: error codes, stack traces, anomaly patterns, "
        "and propose a severity score (0-10). Respond strictly in JSON."
    )

    def __init__(self, agent_id: str | None = None) -> None:
        super().__init__(
            agent_id=agent_id,
            name="LogAnalystAgent",
            model="gemini-2.5-flash",
            system_prompt=self.PERSONA_PROMPT,
            generation_config=GenerationConfig(temperature=0.1, max_output_tokens=2048),
        )

    # ------------------------------------------------------------------
    # SkillSet
    # ------------------------------------------------------------------

    def _build_skillset(self) -> SkillSet:
        parse_skill = _make_skill(
            name="parse_logs",
            description="Ingest raw log text and return a structured anomaly report.",
            parameters=[
                SkillParameter(name="log_text", type="string", description="Raw log content"),
                SkillParameter(name="source_id", type="string", description="Identifier for the log origin"),
            ],
        )
        triage_skill = _make_skill(
            name="triage_anomaly",
            description="Given an anomaly report, assign routing priority and remediation hints.",
            parameters=[
                SkillParameter(name="report", type="object", description="Anomaly report dict"),
            ],
        )

        self.register_skill(AgentSkillDescriptor(skill=parse_skill, handler=self._skill_parse_logs, description="Log parser"))
        self.register_skill(AgentSkillDescriptor(skill=triage_skill, handler=self._skill_triage, description="Anomaly triage"))

        return SkillSet(skills=[parse_skill, triage_skill])

    # ------------------------------------------------------------------
    # Skill handlers
    # ------------------------------------------------------------------

    async def _skill_parse_logs(self, log_text: str, source_id: str) -> dict[str, Any]:
        prompt = f"SOURCE: {source_id}\n\nLOGS:\n{log_text}"
        response = await self.generate_async(prompt)
        raw = response.text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Graceful fallback: extract JSON-ish block if model wraps in markdown.
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else {"raw": raw}

    async def _skill_triage(self, report: dict[str, Any]) -> dict[str, Any]:
        severity = report.get("severity_score", 5)
        priority = min(10, max(0, int(severity)))
        return {
            "priority": priority,
            "route_to": "RemediationAgent" if priority >= 7 else "ObserverAgent",
            "hints": report.get("anomaly_patterns", []),
        }

    # ------------------------------------------------------------------
    # Task processor
    # ------------------------------------------------------------------

    async def _process_task(self, task: AgentTask) -> Any:
        skill_name = task.payload.get("skill")
        descriptor = self.get_skill(skill_name) if skill_name else None

        if descriptor is None:
            # Default: treat entire payload as a raw log parse request.
            return await self._skill_parse_logs(
                log_text=task.payload.get("log_text", ""),
                source_id=task.payload.get("source_id", task.task_id),
            )

        kwargs = {k: v for k, v in task.payload.items() if k != "skill"}
        return await descriptor.handler(**kwargs)


# ---------------------------------------------------------------------------
# DecisionCoordinatorAgent
# ---------------------------------------------------------------------------

class DecisionCoordinatorAgent(BaseAgent):
    """
    Persona: synthesizes diagnostic outputs from peer agents and produces
    coordination decisions — task assignments, escalation signals, retry policies.

    Model: gemini-2.5-pro  (deeper reasoning for multi-source synthesis)
    """

    PERSONA_PROMPT = (
        "You are a distributed systems coordinator. "
        "You receive structured reports from specialist agents and must decide: "
        "which agent should handle the next step, what parameters to pass, "
        "and whether escalation is required. Respond strictly in JSON."
    )

    def __init__(self, agent_id: str | None = None) -> None:
        super().__init__(
            agent_id=agent_id,
            name="DecisionCoordinatorAgent",
            model="gemini-2.5-pro",
            system_prompt=self.PERSONA_PROMPT,
            generation_config=GenerationConfig(temperature=0.3, max_output_tokens=4096),
        )
        self._peer_registry: dict[str, str] = {}  # agent_id → role

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def register_peer(self, agent_id: str, role: str) -> None:
        self._peer_registry[agent_id] = role
        logger.debug("[%s] registered peer %s as '%s'", self.agent_id, agent_id, role)

    # ------------------------------------------------------------------
    # SkillSet
    # ------------------------------------------------------------------

    def _build_skillset(self) -> SkillSet:
        synthesize_skill = _make_skill(
            name="synthesize_reports",
            description="Merge multiple agent reports into a unified decision.",
            parameters=[
                SkillParameter(name="reports", type="array", description="List of structured agent reports"),
            ],
        )
        assign_skill = _make_skill(
            name="assign_task",
            description="Select the optimal agent for a given task description.",
            parameters=[
                SkillParameter(name="task_description", type="string", description="Natural-language task spec"),
                SkillParameter(name="available_agents", type="array", description="List of agent role strings"),
            ],
        )

        self.register_skill(AgentSkillDescriptor(skill=synthesize_skill, handler=self._skill_synthesize, description="Report synthesis"))
        self.register_skill(AgentSkillDescriptor(skill=assign_skill, handler=self._skill_assign, description="Task assignment"))

        return SkillSet(skills=[synthesize_skill, assign_skill])

    # ------------------------------------------------------------------
    # Skill handlers
    # ------------------------------------------------------------------

    async def _skill_synthesize(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        serialized = json.dumps(reports, indent=2)
        prompt = f"Synthesize the following agent reports and decide the next action:\n{serialized}"
        response = await self.generate_async(prompt)
        raw = response.text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else {"raw": raw}

    async def _skill_assign(self, task_description: str, available_agents: list[str]) -> dict[str, Any]:
        agents_str = ", ".join(available_agents)
        prompt = (
            f"Task: {task_description}\n"
            f"Available agents: {agents_str}\n"
            "Select the best agent and specify parameters."
        )
        response = await self.generate_async(prompt)
        raw = response.text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else {"raw": raw}

    # ------------------------------------------------------------------
    # Task processor
    # ------------------------------------------------------------------

    async def _process_task(self, task: AgentTask) -> Any:
        skill_name = task.payload.get("skill")
        descriptor = self.get_skill(skill_name) if skill_name else None

        if descriptor is None:
            # Default: synthesize whatever reports are in payload.
            reports = task.payload.get("reports", [task.payload])
            return await self._skill_synthesize(reports=reports)

        kwargs = {k: v for k, v in task.payload.items() if k != "skill"}
        return await descriptor.handler(**kwargs)
