from __future__ import annotations
from dataclasses import dataclass
import math
import os
import random
from typing import cast

from qbaf import QBAFramework
from qbaf_ctrbs.gradient import determine_gradient_ctrb

from formallymad.agent import Agent


_Relations = tuple[list[tuple[str, str]], list[tuple[str, str]]]
_StrengthCache = dict[tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]], float]


@dataclass
class _QBAFArguments:
    """Flattened QBAF input: argument list, their initial strengths, and agent-to-tool mapping."""
    args: list[str]
    initial_strengths: list[float]
    mapping: dict[str, str]


@dataclass
class _WinningToolSnapshot:
    """Final strengths and winner metadata extracted from a single QBAF evaluation."""
    final_strength_by_tool_name: dict[str, float]
    winning_tool_name: str
    winning_tool_strength: float
    second_best_tool_strength: float


@dataclass
class _ToolStats:
    """
    Running accumulators for tool and agent outcomes across all Monte Carlo permutations.
    """
    strength_sum_by_tool_name: dict[str, float]
    win_count_by_tool_name: dict[str, int]
    win_margin_sum_by_tool_name: dict[str, float]
    influence_sum_by_agent_id: dict[str, float]
    influence_square_sum_by_agent_id: dict[str, float]
    proposal_win_count_by_agent_id: dict[str, int]
    proposed_tool_name_by_agent_id: dict[str, str]

    @classmethod
    def create(cls, tool_names: list[str], agent_ids: list[str],
               tool_proposals: list[tuple[Agent, str, str]]) -> "_ToolStats":
        """Initialise all accumulators to zero for the given tools and agents."""
        return cls(strength_sum_by_tool_name=dict.fromkeys(tool_names, 0.0),
                   win_count_by_tool_name=dict.fromkeys(tool_names, 0),
                   win_margin_sum_by_tool_name=dict.fromkeys(tool_names, 0.0),
                   influence_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   influence_square_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   proposal_win_count_by_agent_id=dict.fromkeys(agent_ids, 0),
                   proposed_tool_name_by_agent_id={agent.id: tool_name for agent, tool_name, _ in tool_proposals})



def normalize_attribution_strengths(scores: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalize llmSHAP attribution scores to [0, 1] for use as QBAF initial strengths.
    If all scores are equal, returns 0.5 for all agents.
    """
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


class QBAFResolver:
    """
    Resolves competing tool proposals via Monte Carlo QBAF evaluation.
    """
    def __init__(self,
                 agents: list[Agent],
                 *,
                 semantics: str = "QuadraticEnergy_model",
                 semantics_aware: bool = False,
                 monte_carlo_permutations: int = 64,
                 visualize: bool = False,
                 seed: int | None = None):
        """
        Initialise the resolver with a pool of agents and QBAF configuration.
        """
        self._agents = agents
        self._semantics = semantics
        self._semantics_aware = semantics_aware
        self._monte_carlo_permutations = max(1, monte_carlo_permutations)
        self._visualize = visualize
        self._plot_counter = 0
        self._random_shuffler = random.Random(seed)
        self.last_tool_stats: dict[str, dict[str, float]] = {}
        self.final_agent_influence: list[tuple[str, float]] = []
        self.last_agent_stats: list[tuple[str, dict[str, float]]] = []


    def resolve(self, tool_proposals: list[tuple[Agent, str, str]]) -> tuple[str, list[tuple[str, float]]]:
        """
        Run Monte Carlo QBAF resolution and return the winning tool name and agent influence scores.
        """
        if not tool_proposals:
            raise ValueError("tool_proposals must not be empty")
        tool_names = list(dict.fromkeys(tool for _, tool, _ in tool_proposals))
        agent_ids = [agent.id for agent in self._agents]
        stats = _ToolStats.create(tool_names, agent_ids, tool_proposals)
        self._plot_counter = 0
        for _ in range(self._monte_carlo_permutations):
            permuted = list(tool_proposals)
            self._random_shuffler.shuffle(permuted)
            qbaf = self._build_qbaf_for_permutation(permuted)
            snapshot = self._compute_winning_tool_snapshot(qbaf, tool_names)
            self._accumulate_tool_outcomes(stats, snapshot)
            self._accumulate_agent_outcomes(stats, agent_ids, snapshot.winning_tool_name, qbaf)
        self.last_tool_stats = self._build_tool_stats(stats, tool_names, float(self._monte_carlo_permutations))
        winner = max(tool_names, key=self._winner_rank_key)
        self.last_agent_stats = self._build_agent_stats(stats, agent_ids, float(self._monte_carlo_permutations))
        self.final_agent_influence = [(agent_id, s["mean_influence"]) for agent_id, s in self.last_agent_stats]
        return winner, self.final_agent_influence


    def _winner_rank_key(self, tool_name: str) -> tuple[float, float]:
        """
        Sort key for selecting the winner: mean strength first, win rate as tiebreaker.
        """
        tool_stats = self.last_tool_stats[tool_name]
        return tool_stats["mean_strength_over_permutations"], tool_stats["top_rank_win_rate_over_permutations"]


    def _build_qbaf_for_permutation(self, permuted_tool_proposals: list[tuple[Agent, str, str]]) -> QBAFramework:
        """
        Construct a QBAFramework for a single permutation of proposals.
        """
        qbaf_args = self._build_arguments(permuted_tool_proposals)
        atts, supps = self._build_relations(qbaf_args.mapping, permuted_tool_proposals)
        qbaf = QBAFramework(qbaf_args.args, qbaf_args.initial_strengths, atts, supps, semantics=self._semantics)
        if self._visualize:
            self.plot(qbaf)
        return qbaf


    def _build_relations(self, mapping: dict[str, str], permuted_tool_proposals: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Dispatch to the semantics-aware or legacy relation builder based on configuration.
        """
        if self._semantics_aware:
            return self._build_semantics_informed_relations(mapping, permuted_tool_proposals)
        return self._build_relations_legacy(mapping, permuted_tool_proposals)


    def _build_relations_legacy(self, mapping: dict[str, str], tool_proposals: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Build attack/support relations using positional order: each agent supports the first advocate of its tool and attacks one agent per opposing tool.
        """
        atts, supps, first_for_tool = [], [], set()
        agent_order = [agent for agent, _, _ in tool_proposals]
        for i, agent in enumerate(agent_order):
            tool = mapping.get(agent.id)
            if not tool: continue
            if tool not in first_for_tool:
                supps.append((agent.id, tool))
                first_for_tool.add(tool)
            supported = False
            attacked_tools = set()
            for prior in reversed(agent_order[:i]):
                prior_tool = mapping.get(prior.id)
                if not prior_tool: continue
                if prior_tool == tool and not supported:
                    supps.append((agent.id, prior.id))
                    supported = True
                elif prior_tool != tool and prior_tool not in attacked_tools:
                    atts.append((agent.id, prior.id))
                    attacked_tools.add(prior_tool)
        return atts, supps


    def _build_semantics_informed_relations(self, mapping: dict[str, str],
                                            permuted_tool_proposals: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Build attack/support relations by greedily selecting the edge that maximises own-tool strength or minimises opponent-tool strength.
        """
        atts: list[tuple[str, str]] = []
        supps: list[tuple[str, str]] = []
        agent_order = [agent for agent, _, _ in permuted_tool_proposals]
        qbaf_args = self._build_arguments(permuted_tool_proposals)
        strength_cache: _StrengthCache = {}
        for i, agent in enumerate(agent_order):
            own_tool = mapping.get(agent.id)
            if not own_tool:
                continue
            prior_agents = agent_order[:i]
            support_targets = [own_tool] + [prior.id for prior in prior_agents if mapping.get(prior.id) == own_tool]
            best_support = self._best_support_target(agent.id, own_tool, support_targets, atts, supps, qbaf_args, strength_cache)
            if best_support is not None:
                supps.append((agent.id, best_support))
            attack_targets = [prior.id for prior in prior_agents if mapping.get(prior.id) not in (None, own_tool)]
            best_attack = self._best_attack_target(agent.id, attack_targets, mapping, atts, supps, qbaf_args, strength_cache)
            if best_attack is not None:
                atts.append((agent.id, best_attack))
        return atts, supps


    def _best_support_target(self, source_agent_id: str, own_tool: str, support_targets: list[str],
                              atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                              qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> str | None:
        """
        Return the support target that yields the greatest increase in own-tool final strength, or None if no beneficial edge exists.
        """
        best_support_target = None
        best_support_gain = 0.0
        for target in support_targets:
            candidate_edge = (source_agent_id, target)
            if candidate_edge in supps:
                continue
            baseline_strength = self._tool_strength_with_relations(own_tool, atts, supps, qbaf_args, strength_cache)
            candidate_strength = self._tool_strength_with_relations(own_tool, atts, supps + [candidate_edge], qbaf_args, strength_cache)
            support_gain = candidate_strength - baseline_strength
            if support_gain > best_support_gain:
                best_support_gain = support_gain
                best_support_target = target
        return best_support_target


    def _best_attack_target(self, source_agent_id: str, attack_targets: list[str], mapping: dict[str, str],
                             atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                             qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> str | None:
        """
        Return the attack target that causes the greatest reduction in the opponent tool's final strength, or None if no harmful edge exists.
        """
        best_attack_target = None
        best_harm = 0.0
        for target_agent_id in attack_targets:
            opponent_tool = mapping.get(target_agent_id)
            if not opponent_tool:
                continue
            candidate_edge = (source_agent_id, target_agent_id)
            if candidate_edge in atts:
                continue
            baseline = self._tool_strength_with_relations(opponent_tool, atts, supps, qbaf_args, strength_cache)
            candidate = self._tool_strength_with_relations(opponent_tool, atts + [candidate_edge], supps, qbaf_args, strength_cache)
            opponent_harm = baseline - candidate
            if opponent_harm > best_harm:
                best_harm = opponent_harm
                best_attack_target = target_agent_id
        return best_attack_target


    def _tool_strength_with_relations(self, tool_name: str, atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                                       qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> float:
        """
        Evaluate and cache the final strength of a tool under the given attack/support relations.
        """
        cache_key = (tool_name, tuple(sorted(atts)), tuple(sorted(supps)))
        if cache_key in strength_cache:
            return strength_cache[cache_key]
        qbaf = QBAFramework(qbaf_args.args, qbaf_args.initial_strengths, atts, supps, semantics=self._semantics)
        final_strength = cast(float, qbaf.final_strengths.get(tool_name, 0.0))
        strength_cache[cache_key] = final_strength
        return final_strength


    def _compute_winning_tool_snapshot(self, qbaf: QBAFramework, tool_names: list[str]) -> _WinningToolSnapshot:
        """
        Extract final strengths from a solved QBAF and identify the top two tools by strength.
        """
        final_strength_by_tool_name = {tool_name: cast(float, qbaf.final_strengths.get(tool_name, 0.0)) for tool_name in tool_names}
        ranked = sorted(final_strength_by_tool_name.items(), key=lambda item: item[1], reverse=True)
        winning_tool_name, winning_tool_strength = ranked[0]
        second_best_tool_strength = ranked[1][1] if len(ranked) > 1 else winning_tool_strength
        return _WinningToolSnapshot(final_strength_by_tool_name, winning_tool_name, winning_tool_strength, second_best_tool_strength)




    ##################################
    # STATISTICS HELPERS
    ##################################
    def _accumulate_tool_outcomes(self, stats: _ToolStats, snapshot: _WinningToolSnapshot) -> None:
        """
        Add one permutation's tool strengths, win, and margin to the running accumulators.
        """
        for tool_name, tool_strength in snapshot.final_strength_by_tool_name.items():
            stats.strength_sum_by_tool_name[tool_name] += tool_strength
        stats.win_count_by_tool_name[snapshot.winning_tool_name] += 1
        stats.win_margin_sum_by_tool_name[snapshot.winning_tool_name] += snapshot.winning_tool_strength - snapshot.second_best_tool_strength


    def _accumulate_agent_outcomes(self, stats: _ToolStats, agent_ids: list[str], winning_tool_name: str, qbaf: QBAFramework) -> None:
        """
        Add one permutation's gradient contributions and proposal win counts to the running accumulators.
        """
        for agent_id in agent_ids:
            influence = cast(float, determine_gradient_ctrb(winning_tool_name, {agent_id}, qbaf))
            stats.influence_sum_by_agent_id[agent_id] += influence
            stats.influence_square_sum_by_agent_id[agent_id] += influence * influence
            stats.proposal_win_count_by_agent_id[agent_id] += int(stats.proposed_tool_name_by_agent_id.get(agent_id) == winning_tool_name)


    def _build_tool_stats(self, stats: _ToolStats, tool_names: list[str], n: float) -> dict[str, dict[str, float]]:
        """
        Compute final summary statistics for each tool from the accumulated Monte Carlo data.
        """
        return {tool_name: self._tool_stat_entry(stats, tool_name, n) for tool_name in tool_names}

    def _tool_stat_entry(self, stats: _ToolStats, tool_name: str, n: float) -> dict[str, float]:
        """
        Build the stats dict for a single tool: mean strength, win rate, and mean winning margin.
        """
        wins = stats.win_count_by_tool_name[tool_name]
        return {
            "mean_strength_over_permutations": round(stats.strength_sum_by_tool_name[tool_name] / n, 4),
            "top_rank_win_rate_over_permutations": round(wins / n, 4),
            "mean_winning_margin_against_second_best": round(stats.win_margin_sum_by_tool_name[tool_name] / wins, 4) if wins else 0.0,
        }


    def _build_agent_stats(self, stats: _ToolStats, agent_ids: list[str], n: float) -> list[tuple[str, dict[str, float]]]:
        """
        Compute final summary statistics for each agent, sorted by absolute mean influence descending.
        """
        entries = [(agent_id, self._agent_stat_entry(stats, agent_id, n)) for agent_id in agent_ids]
        return sorted(entries, key=lambda pair: abs(pair[1]["mean_influence"]), reverse=True)


    def _agent_stat_entry(self, stats: _ToolStats, agent_id: str, n: float) -> dict[str, float]:
        """
        Build the stats dict for a single agent: mean influence, influence std, and proposal win rate.
        """
        mean_influence = stats.influence_sum_by_agent_id[agent_id] / n
        variance = stats.influence_square_sum_by_agent_id[agent_id] / n - mean_influence ** 2
        return {
            "mean_influence": round(mean_influence, 4),
            "influence_std": round(math.sqrt(max(0.0, variance)), 4),
            "proposal_win_rate": round(stats.proposal_win_count_by_agent_id[agent_id] / n, 4),
        }


    def _build_arguments(self, tool_proposals: list[tuple[Agent, str, str]]) -> _QBAFArguments:
        """
        Flatten agents and proposed tools into a unified argument list with initial strengths and agent-to-tool mapping.
        """
        agent_args = [agent.id for agent in self._agents]
        args = list(dict.fromkeys(agent_args + [tool for _, tool, _ in tool_proposals]))
        strengths = {agent.id: agent.strength for agent in self._agents} | {tool: 0.5 for _, tool, _ in tool_proposals}
        mapping = {agent.id: tool for agent, tool, _ in tool_proposals}
        return _QBAFArguments(args=args, initial_strengths=[strengths[arg] for arg in args], mapping=mapping)






    ##################################
    # PLOT HELPERS
    ##################################
    def plot(self, qbaf: QBAFramework) -> None:
        """
        Render the QBAF with final strengths and save it as a PNG in plots-dir.
        """
        from qbaf_visualizer.Visualizer import visualize
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)
        visualize(qbaf, with_fs=True, round_to=3)
        plt.savefig(f"plots/permutation_{self._plot_counter}.png", dpi=300, bbox_inches="tight")
        plt.close()
        self._plot_counter += 1
