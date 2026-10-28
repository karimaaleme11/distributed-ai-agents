# Distributed Multi-Agent Coordination System

> **Production-grade asynchronous agent orchestration built on the [google-antigravity](https://pypi.org/project/google-antigravity/) SDK, powered by Gemini models.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Component Reference](#component-reference)
   - [BaseAgent](#baseagent)
   - [CustomAgent Personas](#customagent-personas)
   - [AgentCoordinator](#agentcoordinator)
   - [WorkspaceEnvironment](#workspaceenvironment)
   - [AgentGraphVisualizer](#agentgraphvisualizer)
4. [Telemetry Matrices](#telemetry-matrices)
5. [Data Flow & Delegation Protocol](#data-flow--delegation-protocol)
6. [Installation & Quick Start](#installation--quick-start)
7. [Configuration Reference](#configuration-reference)
8. [Extension Points](#extension-points)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AgentCoordinator                         │
│  RoutingRules ──► SubagentHandle.delegate()  ──► EventTrigger   │
│                          │                                      │
│          ┌───────────────┼────────────────────┐                 │
│          ▼               ▼                    ▼                 │
│  LogAnalystAgent  DecisionCoordinatorAgent  (future peers)      │
│  [gemini-2.5-flash] [gemini-2.5-pro]                            │
│   PriorityQueue    PriorityQueue                                │
│   Semaphore(4)     Semaphore(4)                                  │
│          │               │                                      │
│          └───────────────┴──────► WorkspaceEnvironment          │
│                                   Artifact Store                │
│                                   Broadcast Bus                 │
│                                   Agent Channels                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   AgentGraphVisualizer
                   DiGraph + Gantt + Dashboard
```

The system is built on three orthogonal concerns:

| Layer | Responsibility |
|---|---|
| **Agent** | Autonomous task execution, skill invocation, Gemini model interaction |
| **Coordination** | Task routing, subagent delegation, event-driven triggering |
| **Simulation** | Shared workspace state, artifact versioning, telemetry visualization |

All inter-agent communication is **non-blocking** (async/await throughout) and event-sourced via the `WorkspaceEnvironment` event log.

---

## Directory Structure

```
distributed-ai-agents/
│
├── requirements.txt               # Pinned runtime dependencies
│
├── agents/
│   ├── __init__.py                # Public surface: BaseAgent, CustomAgent, AgentCoordinator
│   ├── base_agent.py              # Abstract SDK wrapper + async task queue primitives
│   ├── custom_agent.py            # Gemini-backed operational personas + SkillSet definitions
│   └── coordinator.py             # Distributed routing, SubagentHandle delegation, triggers
│
└── simulation/
    ├── __init__.py                # Public surface: WorkspaceEnvironment, AgentGraphVisualizer
    ├── environment.py             # Versioned artifact store + agent channel bus
    └── visualizer.py             # Live coordination graph + Gantt timeline dashboard
```

---

## Component Reference

### BaseAgent

**File:** `agents/base_agent.py`  
**Extends:** `google.antigravity.Agent`

The abstract base layer for every autonomous agent in the system. It wires together the Antigravity SDK lifecycle with deterministic, back-pressure-aware task execution.

#### Core Primitives

| Primitive | Implementation | Purpose |
|---|---|---|
| Task queue | `asyncio.PriorityQueue[AgentTask]` | Priority-ordered task dispatch |
| Concurrency bound | `asyncio.Semaphore(MAX_CONCURRENT_TASKS=4)` | Prevent agent overload |
| Skill registry | `dict[str, AgentSkillDescriptor]` | Named callable skills exposed to runtime |
| Task history | `list[AgentTask]` | Ordered, immutable execution log |

#### `AgentTask` State Machine

```
  enqueue()
     │
     ▼
  PENDING ──► RUNNING ──► COMPLETE
                  │
                  └──────► FAILED
```

Each `AgentTask` carries: `task_id` (UUID4), `name`, `payload: dict`, `priority: int`, `state: TaskState`, `result`, `error`, and timestamps.

#### Key Abstract Methods

```python
@abstractmethod
def _build_skillset(self) -> SkillSet: ...

@abstractmethod
async def _process_task(self, task: AgentTask) -> Any: ...
```

#### Lifecycle

```python
await agent.start()    # Launches background _worker_loop asyncio.Task
await agent.enqueue(task)
await agent.stop()     # Drains queue, then cancels worker
```

---

### CustomAgent Personas

**File:** `agents/custom_agent.py`

Two concrete persona implementations, each with a distinct model, system prompt, generation config, and SkillSet.

#### `LogAnalystAgent`

| Property | Value |
|---|---|
| **Model** | `gemini-2.5-flash` |
| **Temperature** | `0.1` (deterministic extraction) |
| **Max tokens** | `2048` |
| **Skills** | `parse_logs`, `triage_anomaly` |

**System prompt intent:** High-throughput log ingestion → structured JSON anomaly reports with severity scores (0–10).

**`parse_logs` skill flow:**
```
raw log text + source_id
       │
       ▼
  generate_async(prompt)
       │
       ▼
  JSON parse (with markdown-fence fallback via re.search)
       │
       ▼
  { error_codes, stack_traces, anomaly_patterns, severity_score }
```

**`triage_anomaly` skill flow:**
```
anomaly_report
       │
       ▼
  severity_score → priority (0–10)
       │
  priority ≥ 7 → route_to: RemediationAgent
  priority  < 7 → route_to: ObserverAgent
```

#### `DecisionCoordinatorAgent`

| Property | Value |
|---|---|
| **Model** | `gemini-2.5-pro` |
| **Temperature** | `0.3` (balanced reasoning) |
| **Max tokens** | `4096` |
| **Skills** | `synthesize_reports`, `assign_task` |

**System prompt intent:** Multi-source agent report synthesis → JSON coordination decisions (agent assignment, escalation signals, retry policies).

**`synthesize_reports` skill:** Serializes `n` peer reports as JSON, submits to Gemini Pro, parses structured decision payload.

**`assign_task` skill:** Given a natural-language task spec and a list of available agent roles, emits a JSON assignment with parameters.

---

### AgentCoordinator

**File:** `agents/coordinator.py`

The decentralized routing and delegation layer. Does **not** extend `BaseAgent` — it is a pure orchestration controller.

#### Routing Pipeline

```
dispatch(task)
     │
     ▼
_find_rule(task)  ─── RoutingRule.pattern ∈ task.name ?
     │                        │ yes
     │                        ▼
     │              apply priority_override + transform()
     │
     ▼
_resolve_target()
     │
     ├── agent.subagent_handle is not None?
     │        └──► SubagentHandle.delegate(task_id, payload, priority)
     │
     └── else ──► agent.enqueue(task)
     │
     ▼
DelegationRecord appended to _delegation_log
     │
     ▼
_message_bus.put({ event, task_id, task_name, target_agent_id })
     │
     ▼
_evaluate_triggers()  ──► EventTrigger.condition.matches(msg)?
                                  └──► trigger.callback(msg)
```

#### Key Types

```python
@dataclass
class RoutingRule:
    pattern: str                        # substring match on task.name
    target_agent_id: str
    priority_override: int | None
    transform: Callable[[AgentTask], AgentTask] | None

@dataclass
class DelegationRecord:
    delegated_at: datetime
    source_agent_id: str
    target_agent_id: str
    task_id: str
    task_name: str
    state: TaskState
```

#### `broadcast()` Fan-out

Fans a single task out to **all** registered agents concurrently using `asyncio.gather`, returning the list of target agent IDs that accepted.

---

### WorkspaceEnvironment

**File:** `simulation/environment.py`

Shared mutable state container. Thread-safe under the asyncio event loop.

#### Subsystems

| Subsystem | Backing type | API |
|---|---|---|
| Artifact store | `dict[id → list[Artifact]]` | `put_artifact`, `get_artifact(id, version=-1)`, `list_artifacts(kind?)` |
| Agent channels | `dict[id → asyncio.Queue]` | `send_to(agent_id, msg)`, `receive(agent_id)` |
| Broadcast bus | `asyncio.Queue` + subscriber queues | `broadcast(msg)`, `subscribe(agent_id)` |
| Event log | `list[EnvironmentEvent]` | `event_log` property |

#### `Artifact` versioning

Every `put_artifact()` call **appends** a new version rather than overwriting, enabling full replay:

```python
artifact = env.get_artifact("my-report", version=0)   # first version
artifact = env.get_artifact("my-report", version=-1)  # latest (default)
```

Content integrity is verified via SHA-256 truncated to 12 hex chars (`artifact.content_hash`).

#### `run_workflow()`

Executes an ordered list of `steps` sequentially, resolving each step's `agent_role` from a caller-supplied `agent_map`:

```python
steps = [
    {"task_name": "parse_logs",       "agent_role": "log_analyst", "payload": {...}},
    {"task_name": "synthesize_reports","agent_role": "coordinator", "payload": {...}},
]
results = await env.run_workflow(steps, agent_map={"log_analyst": analyst, "coordinator": dec})
```

---

### AgentGraphVisualizer

**File:** `simulation/visualizer.py`  
**Dependencies:** `networkx`, `matplotlib`

Live coordination telemetry renderer. Maintains an internal `nx.DiGraph` that is mutated incrementally as agents and delegations are recorded.

#### Render Modes

| Method | Output |
|---|---|
| `render_graph(layout, show, save_path)` | Role-colored node graph with edge-weight scaling |
| `render_timeline(show, save_path)` | Gantt-style horizontal bar chart per agent |
| `render_dashboard(show, save_path)` | 2×2 composite: graph + timeline + pie chart + bar chart |

#### Node Color Palette

| Role | Hex |
|---|---|
| `coordinator` | `#E74C3C` |
| `log_analyst` | `#3498DB` |
| `decision` | `#2ECC71` |
| `observer` | `#9B59B6` |
| `remediation` | `#F39C12` |

#### Graph Layouts

`spring` (default, seed=42) · `shell` · `spectral` · `kamada_kawai` (fallback)

#### `graph_metrics()` output schema

```json
{
  "nodes": 3,
  "edges": 5,
  "density": 0.8333,
  "is_dag": false,
  "avg_out_degree": 1.67
}
```

---

## Telemetry Matrices

### Task Execution Telemetry

| Metric | Source | Field |
|---|---|---|
| Task throughput | `BaseAgent._task_history` | count by `TaskState` |
| Latency per task | `AgentTask.completed_at - created_at` | `timedelta` |
| Queue depth | `BaseAgent.pending_count` | `int` |
| Concurrency ceiling | `BaseAgent.MAX_CONCURRENT_TASKS` | `int` (default 4) |
| Failure rate | `sum(state=FAILED) / len(history)` | `float` |

### Delegation Telemetry

| Metric | Source | Field |
|---|---|---|
| Total delegations | `len(coordinator.delegation_log())` | `int` |
| Edge frequency | `DiGraph[u][v]["weight"]` | `int` |
| Out-degree per agent | `DiGraph.out_degree(agent_id)` | `int` |
| Graph density | `nx.density(graph)` | `float ∈ [0,1]` |
| DAG status | `nx.is_directed_acyclic_graph(graph)` | `bool` |

### Environment Telemetry

| Metric | Source | Field |
|---|---|---|
| Artifact versions | `sum(len(v) for v in _artifacts.values())` | `int` |
| Unique artifacts | `len(_artifacts)` | `int` |
| Event log depth | `len(_event_log)` | `int` |
| Active channels | `len(_channels)` | `int` |
| Broadcast subscriber count | `sum(len(qs) for qs in _subscribers.values())` | `int` |

---

## Data Flow & Delegation Protocol

```
External caller
    │
    ▼  AgentTask(name="parse_logs", payload={log_text, source_id})
AgentCoordinator.dispatch()
    │
    ├─ RoutingRule("parse_logs") → LogAnalystAgent
    │
    ▼
LogAnalystAgent._worker_loop()
    │
    ├─ _execute(task) [semaphore-bounded]
    │        │
    │        ▼
    │   _process_task() → _skill_parse_logs()
    │        │
    │        ▼
    │   generate_async(prompt) → Gemini Flash API
    │        │
    │        ▼
    │   JSON parse → anomaly_report
    │
    ├─ task.state = COMPLETE
    ├─ _task_history.append(task)
    │
    ▼
DelegationRecord(state=COMPLETE)  →  coordinator._delegation_log
    │
    ▼
message_bus.put({event: "task_dispatched", ...})
    │
    ▼
_evaluate_triggers() → matched EventTrigger.callback()
    │
    ▼
[optional] follow-up AgentTask dispatched to DecisionCoordinatorAgent
```

---

## Installation & Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/karimaaleme11/distributed-ai-agents.git
cd distributed-ai-agents

# 2. Create isolated environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure Gemini API key
export GOOGLE_API_KEY="your-api-key-here"
```

### Minimal Usage Example

```python
import asyncio
from agents import BaseAgent, CustomAgent, AgentCoordinator
from agents.base_agent import AgentTask
from agents.coordinator import RoutingRule
from agents.custom_agent import LogAnalystAgent, DecisionCoordinatorAgent
from simulation import WorkspaceEnvironment, AgentGraphVisualizer

async def main():
    # Instantiate agents
    analyst = LogAnalystAgent()
    decision = DecisionCoordinatorAgent()

    # Wire coordinator
    coordinator = AgentCoordinator()
    coordinator.register_agent(analyst, role="log_analyst")
    coordinator.register_agent(decision, role="coordinator")
    coordinator.add_routing_rule(RoutingRule(
        pattern="parse_logs",
        target_agent_id=analyst.agent_id,
        priority_override=8,
    ))

    # Shared workspace
    env = WorkspaceEnvironment(workspace_id="demo")
    env.register_agent_channel(analyst.agent_id)

    # Visualizer
    viz = AgentGraphVisualizer(title="Demo Run")
    viz.add_agent(analyst.agent_id, role="log_analyst", label="LogAnalyst")
    viz.add_agent(decision.agent_id, role="decision", label="Decision")
    viz.add_delegation("coordinator", analyst.agent_id)

    # Start and dispatch
    await coordinator.start()
    task = AgentTask(name="parse_logs", payload={
        "log_text": "ERROR 500 NullPointerException at line 42",
        "source_id": "service-auth",
    })
    await coordinator.dispatch(task)

    # Render dashboard
    await asyncio.sleep(2)
    viz.render_dashboard(save_path="dashboard.png")
    await coordinator.stop()

asyncio.run(main())
```

---

## Configuration Reference

### `BaseAgent` constructor kwargs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str \| None` | UUID4 | Stable identity across restarts |
| `name` | `str` | `"BaseAgent"` | Human-readable label |
| `model` | `str` | `"gemini-2.5-pro"` | Gemini model identifier |
| `MAX_CONCURRENT_TASKS` | `int` (class var) | `4` | Semaphore ceiling |

### `GenerationConfig` per persona

| Agent | `temperature` | `max_output_tokens` | Rationale |
|---|---|---|---|
| `LogAnalystAgent` | `0.1` | `2048` | Deterministic extraction |
| `DecisionCoordinatorAgent` | `0.3` | `4096` | Nuanced multi-step reasoning |

### `RoutingRule` fields

| Field | Type | Semantics |
|---|---|---|
| `pattern` | `str` | Substring match on `AgentTask.name` |
| `target_agent_id` | `str` | UUID of destination agent |
| `priority_override` | `int \| None` | Replaces task priority pre-dispatch |
| `transform` | `Callable[[AgentTask], AgentTask] \| None` | Arbitrary task mutation before enqueue |

---

## Extension Points

### Adding a new agent persona

1. Subclass `BaseAgent`.
2. Implement `_build_skillset()` returning a `SkillSet` with your `Skill` definitions.
3. Implement `_process_task()` dispatching to skill handlers.
4. Register with `AgentCoordinator.register_agent(agent, role="your-role")`.
5. Add a `RoutingRule` mapping task name patterns to your agent's `agent_id`.

### Adding a new trigger

```python
from google.antigravity.triggers import TriggerCondition

condition = TriggerCondition(field="event", value="task_dispatched")
async def on_dispatch(message):
    print(f"Task {message['task_id']} dispatched to {message['target_agent_id']}")

coordinator.register_trigger(condition, on_dispatch)
```

### Adding a new visualization panel

Subclass or extend `AgentGraphVisualizer`, add a `_render_*` method following the existing `ax: plt.Axes` convention, and compose it into `render_dashboard()` via `fig.add_gridspec`.

---

