"""
Concrete agent personas using google-genai (Gemini) for AI reasoning.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types as genai_types

from .base_agent import AgentTask, BaseAgent, SkillDescriptor

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    return genai.Client(api_key=api_key)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, handling markdown code fences."""
    text = text.strip()
    # Strip ```json ... ``` fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"raw_output": text}


# ---------------------------------------------------------------------------
# LogAnalystAgent
# ---------------------------------------------------------------------------

class LogAnalystAgent(BaseAgent):
    """
    Reads raw log streams with Gemini Flash and returns structured anomaly reports.
    Model: gemini-2.5-flash  (fast, high-throughput)
    """

    SYSTEM_PROMPT = (
        "You are a precision log-analysis engine. "
        "Given raw log lines, extract: error_codes, stack_traces, anomaly_patterns, "
        "and a severity_score (0-10). Always respond with valid JSON only, no markdown."
    )

    def __init__(self, agent_id: str | None = None) -> None:
        super().__init__(agent_id=agent_id, name="LogAnalystAgent", model="gemini-3.8-flash")

    def _register_skills(self) -> None:
        self.add_skill(SkillDescriptor(
            name="parse_logs",
            description="Parse raw log text and return a structured anomaly report.",
            handler=self._skill_parse_logs,
        ))
        self.add_skill(SkillDescriptor(
            name="triage_anomaly",
            description="Assign routing priority from an anomaly report.",
            handler=self._skill_triage,
        ))

    async def _skill_parse_logs(self, log_text: str, source_id: str = "unknown") -> dict[str, Any]:
        client = _get_client()
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"SOURCE: {source_id}\n\nLOGS:\n{log_text}"
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048),
        )
        return _extract_json(response.text)

    async def _skill_triage(self, report: dict[str, Any]) -> dict[str, Any]:
        severity = int(report.get("severity_score", 5))
        priority = min(10, max(0, severity))
        return {
            "priority":  priority,
            "route_to":  "RemediationAgent" if priority >= 7 else "ObserverAgent",
            "hints":     report.get("anomaly_patterns", []),
        }

    async def _process_task(self, task: AgentTask) -> Any:
        skill_name = task.payload.get("skill")
        descriptor = self.get_skill(skill_name) if skill_name else None
        if descriptor is None:
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
    Synthesizes peer-agent reports into coordination decisions with Gemini Pro.
    Model: gemini-2.5-pro  (deep multi-source reasoning)
    """

    SYSTEM_PROMPT = (
        "You are a distributed systems coordinator. "
        "You receive structured JSON reports from specialist agents and must decide: "
        "which agent handles the next step, what parameters to pass, "
        "and whether escalation is required. Respond with valid JSON only, no markdown."
    )

    def __init__(self, agent_id: str | None = None) -> None:
        super().__init__(agent_id=agent_id, name="DecisionCoordinatorAgent", model="gemini-3.1-pro-preview")
        self._peer_registry: dict[str, str] = {}

    def register_peer(self, agent_id: str, role: str) -> None:
        self._peer_registry[agent_id] = role

    def _register_skills(self) -> None:
        self.add_skill(SkillDescriptor(
            name="synthesize_reports",
            description="Merge multiple agent reports into a unified decision.",
            handler=self._skill_synthesize,
        ))
        self.add_skill(SkillDescriptor(
            name="assign_task",
            description="Select the optimal agent for a given task description.",
            handler=self._skill_assign,
        ))

    async def _skill_synthesize(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
        client = _get_client()
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Synthesize the following agent reports and decide the next action:\n"
            f"{json.dumps(reports, indent=2)}"
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=4096),
        )
        return _extract_json(response.text)

    async def _skill_assign(self, task_description: str, available_agents: list[str]) -> dict[str, Any]:
        client = _get_client()
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Task: {task_description}\n"
            f"Available agents: {', '.join(available_agents)}\n"
            "Select the best agent and specify parameters."
        )
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=2048),
        )
        return _extract_json(response.text)

    async def _process_task(self, task: AgentTask) -> Any:
        skill_name = task.payload.get("skill")
        descriptor = self.get_skill(skill_name) if skill_name else None
        if descriptor is None:
            reports = task.payload.get("reports", [task.payload])
            return await self._skill_synthesize(reports=reports)
        kwargs = {k: v for k, v in task.payload.items() if k != "skill"}
        return await descriptor.handler(**kwargs)
