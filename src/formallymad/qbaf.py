from __future__ import annotations
from typing import cast

from qbaf import QBAFramework
from qbaf_ctrbs.gradient import determine_gradient_ctrb

from formallymad.agent import WorkerAgent


class QBAFResolver:
    def __init__(self, agents: list[WorkerAgent], *, semantics: str = "QuadraticEnergy_model"):
        self.agents, self.semantics = agents, semantics

    def resolve(self, tool_proposals: list[tuple[WorkerAgent, str, str]], *, visualize: bool = False) -> tuple[str, list[tuple[str, float]]]:
        args, initial_strengths, mapping, tools = self._build_arguments(tool_proposals)
        qbaf = QBAFramework(args, initial_strengths, *self._build_relations(mapping), semantics=self.semantics)
        if visualize: self.visualize(qbaf)
        tool_final_strengths: dict[str, float] = {tool: round(qbaf.final_strengths[tool], 2) for tool in tools if tool in qbaf.final_strengths}
        max_tool, max_strength = max(tool_final_strengths.items(), key=lambda tool_and_strength: tool_and_strength[1])
        ctrbs = [(agent.id(), cast(float, determine_gradient_ctrb(max_tool, {agent.id()}, qbaf))) for agent in self.agents]
        return max_tool, ctrbs

    def _build_arguments(self, tool_proposals: list[tuple[WorkerAgent, str, str]]) -> tuple[list[str], list[float], dict[str, str], list[str]]:
        agent_args = [agent.id() for agent in self.agents]
        tools = [tool for _, tool, _ in tool_proposals]
        args = list(dict.fromkeys(agent_args + tools))
        strengths = {agent.id(): agent.strength() for agent in self.agents} | {tool: 0.5 for _, tool, _ in tool_proposals}
        mapping = {agent.id(): tool for agent, tool, _ in tool_proposals}
        return args, [strengths[arg] for arg in args], mapping, tools

    def _build_relations(self, mapping: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        atts, supps, first_for_tool = [], [], set()
        for i, agent in enumerate(self.agents):
            tool = mapping.get(agent.id())
            if not tool: continue
            if tool not in first_for_tool: supps.append((agent.id(), tool)); first_for_tool.add(tool)
            supported, attacked_tools = False, set()
            for prior in reversed(self.agents[:i]):
                prior_tool = mapping.get(prior.id())
                if not prior_tool: continue
                if prior_tool == tool and not supported: supps.append((agent.id(), prior.id())); supported = True
                elif prior_tool != tool and prior_tool not in attacked_tools: atts.append((agent.id(), prior.id())); attacked_tools.add(prior_tool)
        return atts, supps

    @staticmethod
    def visualize(qbaf: QBAFramework, image_path: str = "./qbaf.png") -> None:
        from qbaf_visualizer.Visualizer import visualize
        import matplotlib.pyplot as plt

        visualize(qbaf, with_fs=True, round_to=3)
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close()
