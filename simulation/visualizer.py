"""
Visualization layer for agent interaction graphs and task execution timelines.
Uses networkx for graph topology and matplotlib for rendering.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx

from agents.base_agent import TaskState


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NodeMeta:
    """Visual metadata for a single agent node."""

    agent_id: str
    role: str
    color: str = "#4A90D9"
    size: int = 1800


@dataclass
class EdgeMeta:
    """Tracks message volume on a directed edge."""

    source: str
    target: str
    count: int = 0
    last_seen: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# AgentGraphVisualizer
# ---------------------------------------------------------------------------

class AgentGraphVisualizer:
    """
    Builds and renders a live directed graph of agent interactions.

    Capabilities:
      - add_agent / add_delegation: incrementally update the interaction graph
      - render_graph: draw the topology with weighted edges and role-colored nodes
      - render_timeline: Gantt-style task execution chart per agent
      - render_dashboard: composite figure combining graph + timeline + summary stats
    """

    _ROLE_PALETTE = {
        "coordinator":    "#E74C3C",   # red
        "log_analyst":    "#3498DB",   # blue
        "decision":       "#2ECC71",   # green
        "observer":       "#9B59B6",   # purple
        "remediation":    "#F39C12",   # orange
    }
    _DEFAULT_COLOR = "#95A5A6"

    def __init__(self, title: str = "Agent Interaction Graph") -> None:
        self.title = title
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: dict[str, NodeMeta] = {}
        self._edges: dict[tuple[str, str], EdgeMeta] = {}
        self._task_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Graph mutation
    # ------------------------------------------------------------------

    def add_agent(self, agent_id: str, role: str, label: str | None = None) -> None:
        color = self._ROLE_PALETTE.get(role, self._DEFAULT_COLOR)
        meta = NodeMeta(agent_id=agent_id, role=role, color=color)
        self._nodes[agent_id] = meta
        self._graph.add_node(agent_id, label=label or role, color=color, role=role)

    def add_delegation(self, source_id: str, target_id: str) -> None:
        key = (source_id, target_id)
        if key not in self._edges:
            self._edges[key] = EdgeMeta(source=source_id, target=target_id)
            self._graph.add_edge(source_id, target_id, weight=1)
        else:
            self._edges[key].count += 1
            self._edges[key].last_seen = datetime.utcnow()
            self._graph[source_id][target_id]["weight"] += 1

    def record_task(
        self,
        agent_id: str,
        task_name: str,
        start: datetime,
        end: datetime,
        state: TaskState,
    ) -> None:
        self._task_records.append({
            "agent_id": agent_id,
            "task_name": task_name,
            "start": start,
            "end": end,
            "state": state,
        })

    # ------------------------------------------------------------------
    # Rendering: interaction graph
    # ------------------------------------------------------------------

    def render_graph(
        self,
        ax: plt.Axes | None = None,
        layout: str = "spring",
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        standalone = ax is None
        fig, ax = (plt.subplots(1, 1, figsize=(10, 7)) if standalone else (ax.figure, ax))

        pos = self._compute_layout(layout)

        node_colors = [self._nodes[n].color if n in self._nodes else self._DEFAULT_COLOR
                       for n in self._graph.nodes()]
        node_sizes  = [self._nodes[n].size if n in self._nodes else 1000
                       for n in self._graph.nodes()]

        edge_weights = [self._graph[u][v].get("weight", 1) for u, v in self._graph.edges()]
        max_w = max(edge_weights, default=1)
        edge_widths = [1.0 + 3.0 * (w / max_w) for w in edge_weights]

        nx.draw_networkx_nodes(self._graph, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.92)
        nx.draw_networkx_labels(
            self._graph, pos, ax=ax,
            labels={n: self._graph.nodes[n].get("label", n)[:12] for n in self._graph.nodes()},
            font_size=8, font_color="white", font_weight="bold",
        )
        nx.draw_networkx_edges(
            self._graph, pos, ax=ax,
            width=edge_widths, alpha=0.7,
            edge_color="#ECF0F1",
            arrows=True, arrowsize=18,
            connectionstyle="arc3,rad=0.1",
        )
        edge_labels = {(u, v): d["weight"] for u, v, d in self._graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=7, label_pos=0.35)

        ax.set_facecolor("#1A1A2E")
        ax.set_title(self.title, color="white", fontsize=13, fontweight="bold", pad=12)
        ax.axis("off")

        legend_patches = [
            mpatches.Patch(color=color, label=role)
            for role, color in self._ROLE_PALETTE.items()
            if any(m.role == role for m in self._nodes.values())
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, loc="lower left",
                      framealpha=0.3, facecolor="#2C3E50", labelcolor="white", fontsize=8)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
        if standalone and show:
            plt.show()

        return fig

    # ------------------------------------------------------------------
    # Rendering: Gantt timeline
    # ------------------------------------------------------------------

    def render_timeline(
        self,
        ax: plt.Axes | None = None,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        standalone = ax is None
        fig, ax = (plt.subplots(figsize=(12, 5)) if standalone else (ax.figure, ax))

        if not self._task_records:
            ax.text(0.5, 0.5, "No task records", ha="center", va="center",
                    transform=ax.transAxes, color="white")
        else:
            agent_ids = sorted({r["agent_id"] for r in self._task_records})
            y_pos = {aid: i for i, aid in enumerate(agent_ids)}
            cmap = cm.get_cmap("tab10")
            state_colors = {
                TaskState.COMPLETE: "#2ECC71",
                TaskState.FAILED:   "#E74C3C",
                TaskState.RUNNING:  "#F39C12",
                TaskState.PENDING:  "#95A5A6",
            }

            t0 = min(r["start"] for r in self._task_records)
            for rec in self._task_records:
                y = y_pos[rec["agent_id"]]
                x_start = (rec["start"] - t0).total_seconds()
                duration = max((rec["end"] - rec["start"]).total_seconds(), 0.05)
                color = state_colors.get(rec["state"], "#BDC3C7")
                ax.barh(y, duration, left=x_start, height=0.5, color=color, alpha=0.85, edgecolor="none")
                if duration > 0.5:
                    ax.text(x_start + duration / 2, y, rec["task_name"][:14],
                            va="center", ha="center", fontsize=6, color="white")

            ax.set_yticks(list(y_pos.values()))
            ax.set_yticklabels(list(y_pos.keys()), color="white", fontsize=8)
            ax.set_xlabel("Elapsed seconds", color="white", fontsize=9)
            ax.set_title("Task Execution Timeline", color="white", fontsize=11, fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2C3E50")

        ax.set_facecolor("#1A1A2E")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
        if standalone and show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Rendering: composite dashboard
    # ------------------------------------------------------------------

    def render_dashboard(
        self,
        show: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        fig = plt.figure(figsize=(18, 10), facecolor="#0D0D1A")
        gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3,
                              left=0.06, right=0.97, top=0.92, bottom=0.06)

        ax_graph    = fig.add_subplot(gs[0, 0])
        ax_timeline = fig.add_subplot(gs[0, 1])
        ax_stats    = fig.add_subplot(gs[1, 0])
        ax_degree   = fig.add_subplot(gs[1, 1])

        self.render_graph(ax=ax_graph, show=False)
        self.render_timeline(ax=ax_timeline, show=False)
        self._render_stats(ax_stats)
        self._render_degree_distribution(ax_degree)

        fig.suptitle("Distributed Agent System — Dashboard",
                     color="white", fontsize=15, fontweight="bold", y=0.97)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D0D1A")
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Private sub-plots
    # ------------------------------------------------------------------

    def _render_stats(self, ax: plt.Axes) -> None:
        ax.set_facecolor("#1A1A2E")
        ax.set_title("Task State Distribution", color="white", fontsize=10, fontweight="bold")
        ax.axis("off")

        if not self._task_records:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="white")
            return

        counts: dict[str, int] = collections.Counter(
            r["state"].name for r in self._task_records
        )
        colors = {"COMPLETE": "#2ECC71", "FAILED": "#E74C3C",
                  "RUNNING": "#F39C12", "PENDING": "#95A5A6"}
        labels = list(counts.keys())
        values = list(counts.values())
        wedge_colors = [colors.get(l, "#BDC3C7") for l in labels]

        ax.axis("on")
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, colors=wedge_colors,
            autopct="%1.0f%%", startangle=140,
            textprops={"color": "white", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("white")

    def _render_degree_distribution(self, ax: plt.Axes) -> None:
        ax.set_facecolor("#1A1A2E")
        ax.set_title("Agent Delegation Frequency", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")

        if not self._graph.edges():
            ax.text(0.5, 0.5, "No delegations", ha="center", va="center",
                    transform=ax.transAxes, color="white")
            return

        out_degrees = dict(self._graph.out_degree())
        agents = [self._graph.nodes[n].get("label", n)[:10] for n in out_degrees]
        degrees = list(out_degrees.values())
        cmap = cm.get_cmap("Blues")
        bar_colors = [cmap(0.4 + 0.5 * (d / max(degrees, default=1))) for d in degrees]

        bars = ax.bar(agents, degrees, color=bar_colors, edgecolor="none")
        ax.set_ylabel("Delegations out", color="white", fontsize=8)
        ax.set_xlabel("Agent", color="white", fontsize=8)
        ax.set_xticklabels(agents, rotation=30, ha="right", color="white", fontsize=7)
        ax.set_yticklabels(ax.get_yticks(), color="white")

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _compute_layout(self, layout: str) -> dict[str, Any]:
        if not self._graph.nodes():
            return {}
        if layout == "spring":
            return nx.spring_layout(self._graph, seed=42, k=2.0)
        if layout == "shell":
            return nx.shell_layout(self._graph)
        if layout == "spectral" and len(self._graph) > 2:
            return nx.spectral_layout(self._graph)
        return nx.kamada_kawai_layout(self._graph)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def graph_metrics(self) -> dict[str, Any]:
        g = self._graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": round(nx.density(g), 4),
            "is_dag": nx.is_directed_acyclic_graph(g),
            "avg_out_degree": round(
                sum(d for _, d in g.out_degree()) / max(g.number_of_nodes(), 1), 2
            ),
        }
