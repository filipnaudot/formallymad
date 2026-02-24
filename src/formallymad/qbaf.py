from __future__ import annotations
from dataclasses import dataclass
import math
import random
from typing import cast

from qbaf import QBAFramework
from qbaf_ctrbs.gradient import determine_gradient_ctrb

from formallymad.agent import WorkerAgent


@dataclass
class MonteCarloStats:
    strength_sum_by_tool_name: dict[str, float]
    win_count_by_tool_name: dict[str, int]
    win_margin_sum_by_tool_name: dict[str, float]
    influence_sum_by_agent_id: dict[str, float]
    influence_square_sum_by_agent_id: dict[str, float]
    proposal_win_count_by_agent_id: dict[str, int]
    proposed_tool_name_by_agent_id: dict[str, str]

    @classmethod
    def create(cls, tool_names: list[str], agent_ids: list[str],
               tool_proposals: list[tuple[WorkerAgent, str, str]]) -> "MonteCarloStats":
        return cls(strength_sum_by_tool_name=dict.fromkeys(tool_names, 0.0),
                   win_count_by_tool_name=dict.fromkeys(tool_names, 0),
                   win_margin_sum_by_tool_name=dict.fromkeys(tool_names, 0.0),
                   influence_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   influence_square_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   proposal_win_count_by_agent_id=dict.fromkeys(agent_ids, 0),
                   proposed_tool_name_by_agent_id={agent.id(): tool_name for agent, tool_name, _ in tool_proposals})



class QBAFResolver:
    def __init__(self, agents: list[WorkerAgent], *, semantics: str = "QuadraticEnergy_model", monte_carlo_permutations: int = 64, seed: int | None = None):
        self._agents = agents
        self._semantics = semantics
        self._monte_carlo_permutations = max(1, monte_carlo_permutations)
        self._rng = random.Random(seed)
        self.last_tool_stats: dict[str, dict[str, float]] = {}
        self.last_agent_influence: list[tuple[str, float]] = []
        self.last_agent_stats: list[tuple[str, dict[str, float]]] = []


    def resolve(self, tool_proposals: list[tuple[WorkerAgent, str, str]], *, VISUALIZE: bool = False) -> tuple[str, list[tuple[str, float]]]:
        if not tool_proposals: raise ValueError("tool_proposals must not be empty")
        tool_names, agent_ids = list(dict.fromkeys(tool for _, tool, _ in tool_proposals)), [agent.id() for agent in self._agents]
        stats = MonteCarloStats.create(tool_names, agent_ids, tool_proposals)
        for _ in range(self._monte_carlo_permutations):
            permuted_tool_proposals = list(tool_proposals)
            self._rng.shuffle(permuted_tool_proposals)
            current_qbaf = self._build_qbaf_for_permutation(permuted_tool_proposals, visualize=VISUALIZE)
            final_strength_by_tool_name, winning_tool_name, winning_tool_strength, second_best_tool_strength = self._compute_winning_tool_snapshot(current_qbaf, tool_names)
            self._accumulate_tool_outcomes(stats.strength_sum_by_tool_name, stats.win_count_by_tool_name, stats.win_margin_sum_by_tool_name, final_strength_by_tool_name, winning_tool_name, winning_tool_strength, second_best_tool_strength)
            self._accumulate_agent_outcomes(agent_ids, stats.proposed_tool_name_by_agent_id, stats.proposal_win_count_by_agent_id, stats.influence_sum_by_agent_id, stats.influence_square_sum_by_agent_id, winning_tool_name, current_qbaf)
        permutation_count_as_float = float(self._monte_carlo_permutations)
        self.last_tool_stats = self._build_tool_stats(tool_names, stats.strength_sum_by_tool_name, stats.win_count_by_tool_name, stats.win_margin_sum_by_tool_name, permutation_count_as_float)
        winner_tool_name = max(tool_names, key=lambda tool_name: (self.last_tool_stats[tool_name]["mean_strength_over_permutations"], self.last_tool_stats[tool_name]["top_rank_win_rate_over_permutations"]))
        self.last_agent_stats = self._build_agent_stats(agent_ids, stats.influence_sum_by_agent_id, stats.influence_square_sum_by_agent_id, stats.proposal_win_count_by_agent_id, permutation_count_as_float)
        self.last_agent_influence = [(agent_id, stats["mean_influence"]) for agent_id, stats in self.last_agent_stats]
        return winner_tool_name, self.last_agent_influence


    def _build_qbaf_for_permutation(self, permuted_tool_proposals: list[tuple[WorkerAgent, str, str]], *, visualize: bool) -> QBAFramework:
        args, initial_strengths, mapping, _ = self._build_arguments(permuted_tool_proposals)
        qbaf_for_permutation = QBAFramework(args, initial_strengths, *self._build_relations(mapping, permuted_tool_proposals), semantics=self._semantics)
        if visualize: self.visualize(qbaf_for_permutation)
        return qbaf_for_permutation


    def _compute_winning_tool_snapshot(self, current_qbaf: QBAFramework, tool_names: list[str]) -> tuple[dict[str, float], str, float, float]:
        final_strength_by_tool_name = {tool_name: cast(float, current_qbaf.final_strengths.get(tool_name, 0.0)) for tool_name in tool_names}
        ranked_tool_names_by_strength = sorted(final_strength_by_tool_name.items(), key=lambda tool_and_strength: tool_and_strength[1], reverse=True)
        winning_tool_name, winning_tool_strength = ranked_tool_names_by_strength[0]
        second_best_tool_strength = ranked_tool_names_by_strength[1][1] if len(ranked_tool_names_by_strength) > 1 else winning_tool_strength
        return final_strength_by_tool_name, winning_tool_name, winning_tool_strength, second_best_tool_strength


    def _accumulate_tool_outcomes(self, 
                                  strength_sum_by_tool_name: dict[str, float], 
                                  win_count_by_tool_name: dict[str, int], 
                                  win_margin_sum_by_tool_name: dict[str, float], 
                                  final_strength_by_tool_name: dict[str, float], 
                                  winning_tool_name: str, 
                                  winning_tool_strength: float, 
                                  second_best_tool_strength: float) -> None:
        for tool_name, tool_strength in final_strength_by_tool_name.items():
            strength_sum_by_tool_name[tool_name] += tool_strength
        win_count_by_tool_name[winning_tool_name] += 1
        win_margin_sum_by_tool_name[winning_tool_name] += winning_tool_strength - second_best_tool_strength


    def _accumulate_agent_outcomes(self, agent_ids: list[str], 
                                   proposed_tool_name_by_agent_id: dict[str, str], 
                                   proposal_win_count_by_agent_id: dict[str, int], 
                                   influence_sum_by_agent_id: dict[str, float], 
                                   influence_square_sum_by_agent_id: dict[str, float], 
                                   winning_tool_name: str, current_qbaf: QBAFramework) -> None:
        for agent_id in agent_ids:
            influence = cast(float, determine_gradient_ctrb(winning_tool_name, {agent_id}, current_qbaf))
            influence_sum_by_agent_id[agent_id] += influence
            influence_square_sum_by_agent_id[agent_id] += influence * influence
            proposal_win_count_by_agent_id[agent_id] += int(proposed_tool_name_by_agent_id.get(agent_id) == winning_tool_name)


    def _build_tool_stats(self, tool_names: list[str], 
                          strength_sum_by_tool_name: dict[str, float], 
                          win_count_by_tool_name: dict[str, int], 
                          win_margin_sum_by_tool_name: dict[str, float], 
                          permutation_count_as_float: float) -> dict[str, dict[str, float]]:
        return {
            tool_name: {
                "mean_strength_over_permutations": round(strength_sum_by_tool_name[tool_name] / permutation_count_as_float, 4),
                "top_rank_win_rate_over_permutations": round(win_count_by_tool_name[tool_name] / permutation_count_as_float, 4),
                "mean_winning_margin_against_second_best": round(win_margin_sum_by_tool_name[tool_name] / win_count_by_tool_name[tool_name], 4) 
                if win_count_by_tool_name[tool_name] else 0.0,} 
                for tool_name in tool_names
            }


    def _build_agent_stats(self, agent_ids: list[str], influence_sum_by_agent_id: dict[str, float], influence_square_sum_by_agent_id: dict[str, float], proposal_win_count_by_agent_id: dict[str, int], permutation_count_as_float: float) -> list[tuple[str, dict[str, float]]]:
        return sorted([(agent_id, {"mean_influence": round(influence_sum_by_agent_id[agent_id] / permutation_count_as_float, 4),
                                   "influence_std": round(math.sqrt(max(0.0, influence_square_sum_by_agent_id[agent_id] / permutation_count_as_float - (influence_sum_by_agent_id[agent_id] / permutation_count_as_float) ** 2)), 4),
                                   "proposal_win_rate": round(proposal_win_count_by_agent_id[agent_id] / permutation_count_as_float, 4),
                                   },)
                                   for agent_id in agent_ids],
            key=lambda agent_id_and_stats: abs(agent_id_and_stats[1]["mean_influence"]),
            reverse=True,
        )


    def _build_arguments(self, tool_proposals: list[tuple[WorkerAgent, str, str]]) -> tuple[list[str], list[float], dict[str, str], list[str]]:
        agent_args = [agent.id() for agent in self._agents]
        tools = [tool for _, tool, _ in tool_proposals]
        args = list(dict.fromkeys(agent_args + tools))
        strengths = {agent.id(): agent.strength() for agent in self._agents} | {tool: 0.5 for _, tool, _ in tool_proposals}
        mapping = {agent.id(): tool for agent, tool, _ in tool_proposals}
        return args, [strengths[arg] for arg in args], mapping, tools


    def _build_relations(self, mapping: dict[str, str], tool_proposals: list[tuple[WorkerAgent, str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        atts, supps, first_for_tool = [], [], set()
        agent_order = [agent for agent, _, _ in tool_proposals]
        for i, agent in enumerate(agent_order):
            tool = mapping.get(agent.id())
            if not tool: continue
            if tool not in first_for_tool: supps.append((agent.id(), tool)); first_for_tool.add(tool)
            supported, attacked_tools = False, set()
            for prior in reversed(agent_order[:i]):
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
