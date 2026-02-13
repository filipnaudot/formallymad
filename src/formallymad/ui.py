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


    def banner(self) -> None:
        self.console.print(Panel.fit(f"[bold grey70]Formally MAD[/bold grey70]\n[grey62]PATH: {Path.cwd()}[/grey62]", border_style="grey50",))


    def ask(self) -> str:
        return self.console.input("[bold blue]> [/bold blue]").strip()


    @contextmanager
    def loading(self, message: str):
        with self.console.status(f"[grey62]{message}[/grey62]", spinner="dots"): yield


    def show_proposals(self, proposals: Iterable[tuple[str, str, str]]) -> None:
        table = Table(title="Worker Proposals", expand=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Tool", style="magenta", no_wrap=True)
        table.add_column("Motivation", style="white")
        for agent_id, tool_name, motivation in proposals:
            table.add_row(agent_id, tool_name, motivation or "-")
        self.console.print(table)


    def show_agent_metrics(self, metrics: Iterable[tuple[str, dict[str, float]]]) -> None:
        table = Table(title="Agent Metrics", expand=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Mean Influence", style="yellow", no_wrap=True)
        table.add_column("Influence Std", style="yellow", no_wrap=True)
        table.add_column("Proposal Win Rate", style="yellow", no_wrap=True)
        for agent_id, stats in metrics:
            table.add_row(agent_id, f"{stats['mean_influence']:.4f}", f"{stats['influence_std']:.4f}", f"{stats['proposal_win_rate']:.4f}",)
        self.console.print(table)


    def show_selected_tool(self, tool_name: str) -> None:
        self.console.print(Panel.fit(f"[bold green]Selected tool:[/] [magenta]{tool_name}[/]"))


    def show_tool_result(self, tool_name: str, tool_args: dict, result: dict) -> None:
        self.console.print(Panel.fit(f"[bold]Tool[/]: {tool_name}\n[bold]Args[/]: {tool_args}\n[bold]Result[/]: {result}",
                                     title="Tool Call",
                                     border_style="grey62",
                                     ))


    def show_assistant(self, text: str) -> None:
        self.console.print(Panel(text or "", title="Assistant", border_style="yellow"))
