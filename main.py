"""
Demo runner — shows the full agent coordination loop.
Run: python main.py
"""

import asyncio
import os
from datetime import datetime

from agents.base_agent import AgentTask
from agents.coordinator import AgentCoordinator, RoutingRule
from agents.custom_agent import DecisionCoordinatorAgent, LogAnalystAgent
from simulation.environment import Artifact, ArtifactKind, WorkspaceEnvironment
from simulation.visualizer import AgentGraphVisualizer


async def main() -> None:
    print("=== Distributed Multi-Agent System Demo ===\n")

    # -- 1. Instantiate agents -------------------------------------------------
    analyst = LogAnalystAgent()
    decision = DecisionCoordinatorAgent()
    print(f"[+] LogAnalystAgent      id={analyst.agent_id[:8]}…")
    print(f"[+] DecisionAgent        id={decision.agent_id[:8]}…\n")

    # -- 2. Wire the coordinator -----------------------------------------------
    coordinator = AgentCoordinator()
    coordinator.register_agent(analyst, role="log_analyst")
    coordinator.register_agent(decision, role="coordinator")

    coordinator.add_routing_rule(RoutingRule(
        pattern="parse_logs",
        target_agent_id=analyst.agent_id,
        priority_override=8,
    ))
    coordinator.add_routing_rule(RoutingRule(
        pattern="synthesize",
        target_agent_id=decision.agent_id,
        priority_override=6,
    ))

    # -- 3. Shared workspace ---------------------------------------------------
    env = WorkspaceEnvironment(workspace_id="demo-run")
    env.register_agent_channel(analyst.agent_id)
    env.register_agent_channel(decision.agent_id)

    # -- 4. Visualizer setup ---------------------------------------------------
    viz = AgentGraphVisualizer(title="Agent Coordination — Demo Run")
    viz.add_agent(analyst.agent_id,  role="log_analyst", label="LogAnalyst")
    viz.add_agent(decision.agent_id, role="decision",    label="Decision")
    viz.add_delegation("coordinator", analyst.agent_id)
    viz.add_delegation(analyst.agent_id, decision.agent_id)

    # -- 5. Start the system ---------------------------------------------------
    await coordinator.start()
    print("[*] Coordinator started. Dispatching tasks…\n")

    # -- 6. Dispatch sample tasks ----------------------------------------------
    tasks = [
        AgentTask(
            name="parse_logs",
            payload={
                "log_text": (
                    "2024-01-15 08:42:11 ERROR NullPointerException in AuthService at line 312\n"
                    "2024-01-15 08:42:12 ERROR Database connection timeout after 30s\n"
                    "2024-01-15 08:42:15 WARN  Retry attempt 3/5 for request-id=abc123\n"
                    "2024-01-15 08:42:20 ERROR 500 Internal Server Error on /api/v2/users\n"
                ),
                "source_id": "service-auth-prod",
            },
        ),
        AgentTask(
            name="parse_logs",
            payload={
                "log_text": (
                    "2024-01-15 09:00:01 INFO  Deploy successful — version 3.2.1\n"
                    "2024-01-15 09:00:45 WARN  Memory usage at 87% on node-02\n"
                    "2024-01-15 09:01:00 ERROR OOM killer triggered on pid 4812\n"
                ),
                "source_id": "service-infra-prod",
            },
        ),
        AgentTask(
            name="synthesize_reports",
            payload={
                "skill": "synthesize_reports",
                "reports": [
                    {"source": "service-auth-prod",  "severity_score": 8, "anomaly_patterns": ["NullPointerException", "DB timeout"]},
                    {"source": "service-infra-prod", "severity_score": 7, "anomaly_patterns": ["OOM killer", "high memory"]},
                ],
            },
        ),
    ]

    start_times: dict[str, datetime] = {}
    for task in tasks:
        start_times[task.task_id] = datetime.utcnow()
        target_id = await coordinator.dispatch(task)
        print(f"  → Dispatched '{task.name}' to {target_id[:8]}…")

    # -- 7. Let workers run ----------------------------------------------------
    print("\n[*] Waiting for agents to process tasks…")
    await asyncio.sleep(5)

    # -- 8. Store a demo artifact ----------------------------------------------
    env.put_artifact(Artifact(
        artifact_id="demo-report-v1",
        kind=ArtifactKind.REPORT,
        owner_agent_id=analyst.agent_id,
        content={"summary": "Demo run complete", "tasks_dispatched": len(tasks)},
        tags=["demo", "v1"],
    ))

    # -- 9. Record task history for timeline -----------------------------------
    for agent in [analyst, decision]:
        for t in agent.task_history:
            if t.completed_at:
                viz.record_task(
                    agent_id=agent.agent_id,
                    task_name=t.name,
                    start=t.created_at,
                    end=t.completed_at,
                    state=t.state,
                )

    # -- 10. Print summary -----------------------------------------------------
    print("\n=== Agent Summary ===")
    for s in coordinator.agent_summary():
        print(f"  {s['name']} [{s['role']}]")
        print(f"    history : {s['history']}")
        print(f"    skills  : {s['skills']}")

    print("\n=== Graph Metrics ===")
    print(" ", viz.graph_metrics())

    print("\n=== Workspace Snapshot ===")
    print(" ", env.snapshot())

    # -- 11. Render dashboard --------------------------------------------------
    print("\n[*] Rendering dashboard → dashboard.png")
    viz.render_dashboard(show=False, save_path="dashboard.png")
    print("[✓] dashboard.png saved in project root\n")

    # -- 12. Teardown ----------------------------------------------------------
    await coordinator.stop()
    print("[✓] Done.")


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  GOOGLE_API_KEY not set. Set it with:")
        print('   export GOOGLE_API_KEY="your-key-here"')
        raise SystemExit(1)

    asyncio.run(main())
