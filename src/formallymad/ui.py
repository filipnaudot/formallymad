from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class FormallyMADUI:
    def __init__(self) -> None:
        self.console = Console()


    def banner(self, text: str) -> None:
        self.console.print(Panel.fit(f"[bold grey70]{text}[/bold grey70]\n[grey62]PATH: {Path.cwd()}[/grey62]", border_style="grey50",))


    def ask(self) -> str:
        return self.console.input("[bold blue]> [/bold blue]").strip()


    @contextmanager
    def loading(self, message: str):
        with self.console.status(f"[grey62]{message}[/grey62]", spinner="dots"): yield


    def show_proposals(self, proposals: Iterable[tuple[str, str, str]]) -> None:
        table = Table(title="Worker Proposals", expand=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Recommendation", style="magenta")
        table.add_column("Motivation", style="white")
        for agent_id, recommendation, motivation in proposals:
            table.add_row(agent_id, recommendation or "-", motivation or "-")
        self.console.print(table)


    def show_agent_metrics(self, metrics: Iterable[tuple[str, dict[str, float]]], strength_by_agent_id: dict[str, float] | None = None) -> None:
        table = Table(title="Agent Metrics", expand=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        if strength_by_agent_id is not None:
            table.add_column("Strength", style="yellow", no_wrap=True)
        table.add_column("Mean Influence", style="yellow", no_wrap=True)
        table.add_column("Influence Std", style="yellow", no_wrap=True)
        table.add_column("Proposal Win Rate", style="yellow", no_wrap=True)
        for agent_id, stats in metrics:
            row = [agent_id]
            if strength_by_agent_id is not None:
                strength = strength_by_agent_id.get(agent_id)
                row.append(f"{strength:.4f}" if strength is not None else "-")
            row += [f"{stats['mean_influence']:.4f}", f"{stats['influence_std']:.4f}", f"{stats['proposal_win_rate']:.4f}"]
            table.add_row(*row)
        self.console.print(table)


    def show_result(self, label: str, value: str, color: str = "magenta") -> None:
        self.console.print(Panel.fit(f"[white]{label}:[/] [{color}]{value}[/]"))


    def show_assistant(self, text: str) -> None:
        self.console.print(Panel(text or "", title="Assistant", border_style="yellow"))
