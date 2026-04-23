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
    """Flattened QBAF input: argument list, their initial strengths, and agent-to-option mapping."""
    args: list[str]
    initial_strengths: list[float]
    mapping: dict[str, str]


@dataclass
class _WinnerSnapshot:
    """Final strengths and winner metadata extracted from a single QBAF evaluation."""
    final_strength_by_option: dict[str, float]
    winner: str
    winner_strength: float
    second_best_strength: float


@dataclass
class _Stats:
    """
    Running accumulators for option and agent outcomes across all Monte Carlo permutations.
    """
    strength_sum_by_option: dict[str, float]
    win_count_by_option: dict[str, int]
    win_margin_sum_by_option: dict[str, float]
    influence_sum_by_agent_id: dict[str, float]
    influence_square_sum_by_agent_id: dict[str, float]
    recommendation_win_count_by_agent_id: dict[str, int]
    recommendation_by_agent_id: dict[str, str]

    @classmethod
    def create(cls, options: list[str], agent_ids: list[str],
               agent_recommendations: list[tuple[Agent, str, str]]) -> "_Stats":
        """Initialise all accumulators to zero for the given options and agents."""
        return cls(strength_sum_by_option=dict.fromkeys(options, 0.0),
                   win_count_by_option=dict.fromkeys(options, 0),
                   win_margin_sum_by_option=dict.fromkeys(options, 0.0),
                   influence_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   influence_square_sum_by_agent_id=dict.fromkeys(agent_ids, 0.0),
                   recommendation_win_count_by_agent_id=dict.fromkeys(agent_ids, 0),
                   recommendation_by_agent_id={agent.id: option for agent, option, _ in agent_recommendations})



def normalize_attribution_strengths(scores: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalize llmSHAP attribution scores to [0, 1] for use as QBAF initial strengths.
    If all scores are equal, returns 0.5 for all agents.
    """
    values = list(scores.values())
    # TODO: consider proportional normalization instead: score / sum(scores).
    #       It avoids collapsing the minimum agent to 0 and does not artificially stretch the range.
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        return {k: 0.5 for k in scores}
    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


class QBAFResolver:
    """
    Resolves competing agent recommendations via Monte Carlo QBAF evaluation.
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
        self.last_option_stats: dict[str, dict[str, float]] = {}
        self.final_agent_influence: list[tuple[str, float]] = []
        self.last_agent_stats: list[tuple[str, dict[str, float]]] = []


    def resolve(self, agent_recommendations: list[tuple[Agent, str, str]]) -> tuple[str, list[tuple[str, float]]]:
        """
        Run Monte Carlo QBAF resolution and return the winning option and agent influence scores.
        """
        if not agent_recommendations:
            raise ValueError("agent_recommendations must not be empty")
        options = list(dict.fromkeys(option for _, option, _ in agent_recommendations))
        agent_ids = [agent.id for agent in self._agents]
        stats = _Stats.create(options, agent_ids, agent_recommendations)
        self._plot_counter = 0
        for _ in range(self._monte_carlo_permutations):
            permuted = list(agent_recommendations)
            self._random_shuffler.shuffle(permuted)
            qbaf = self._build_qbaf_for_permutation(permuted)
            snapshot = self._compute_winner_snapshot(qbaf, options)
            self._accumulate_option_outcomes(stats, snapshot)
            self._accumulate_agent_outcomes(stats, agent_ids, snapshot.winner, qbaf)
        self.last_option_stats = self._build_option_stats(stats, options, float(self._monte_carlo_permutations))
        winner = max(options, key=self._winner_rank_key)
        self.last_agent_stats = self._build_agent_stats(stats, agent_ids, float(self._monte_carlo_permutations))
        self.final_agent_influence = [(agent_id, s["mean_influence"]) for agent_id, s in self.last_agent_stats]
        return winner, self.final_agent_influence


    def _winner_rank_key(self, option: str) -> tuple[float, float]:
        """
        Sort key for selecting the winner: mean strength first, win rate as tiebreaker.
        """
        option_stats = self.last_option_stats[option]
        return option_stats["mean_strength_over_permutations"], option_stats["top_rank_win_rate_over_permutations"]


    def _build_qbaf_for_permutation(self, permuted_recommendations: list[tuple[Agent, str, str]]) -> QBAFramework:
        """
        Construct a QBAFramework for a single permutation of agent recommendations.
        """
        qbaf_args = self._build_arguments(permuted_recommendations)
        atts, supps = self._build_relations(qbaf_args.mapping, permuted_recommendations)
        qbaf = QBAFramework(qbaf_args.args, qbaf_args.initial_strengths, atts, supps, semantics=self._semantics)
        if self._visualize:
            self.plot(qbaf)
        return qbaf


    def _build_relations(self, mapping: dict[str, str], permuted_recommendations: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Dispatch to the semantics-aware or legacy relation builder based on configuration.
        """
        if self._semantics_aware:
            return self._build_semantics_informed_relations(mapping, permuted_recommendations)
        return self._build_relations_legacy(mapping, permuted_recommendations)


    def _build_relations_legacy(self, mapping: dict[str, str], agent_recommendations: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Build attack/support relations using positional order: each agent supports the first advocate of its option and attacks one agent per opposing option.
        """
        atts, supps, first_for_option = [], [], set()
        agent_order = [agent for agent, _, _ in agent_recommendations]
        for i, agent in enumerate(agent_order):
            option = mapping.get(agent.id)
            if not option: continue
            if option not in first_for_option:
                supps.append((agent.id, option))
                first_for_option.add(option)
            supported = False
            attacked_options = set()
            for prior in reversed(agent_order[:i]):
                prior_option = mapping.get(prior.id)
                if not prior_option: continue
                if prior_option == option and not supported:
                    supps.append((agent.id, prior.id))
                    supported = True
                elif prior_option != option and prior_option not in attacked_options:
                    atts.append((agent.id, prior.id))
                    attacked_options.add(prior_option)
        return atts, supps


    def _build_semantics_informed_relations(self, mapping: dict[str, str],
                                            permuted_recommendations: list[tuple[Agent, str, str]]) -> _Relations:
        """
        Build attack/support relations by greedily selecting the edge that maximises own-option strength or minimises opponent-option strength.
        """
        atts: list[tuple[str, str]] = []
        supps: list[tuple[str, str]] = []
        agent_order = [agent for agent, _, _ in permuted_recommendations]
        qbaf_args = self._build_arguments(permuted_recommendations)
        strength_cache: _StrengthCache = {}
        for i, agent in enumerate(agent_order):
            own_option = mapping.get(agent.id)
            if not own_option:
                continue
            prior_agents = agent_order[:i]
            support_targets = [own_option] + [prior.id for prior in prior_agents if mapping.get(prior.id) == own_option]
            best_support = self._best_support_target(agent.id, own_option, support_targets, atts, supps, qbaf_args, strength_cache)
            if best_support is not None:
                supps.append((agent.id, best_support))
            attack_targets = [prior.id for prior in prior_agents if mapping.get(prior.id) not in (None, own_option)]
            best_attack = self._best_attack_target(agent.id, attack_targets, mapping, atts, supps, qbaf_args, strength_cache)
            if best_attack is not None:
                atts.append((agent.id, best_attack))
        return atts, supps


    def _best_support_target(self, source_agent_id: str, own_option: str, support_targets: list[str],
                              atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                              qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> str | None:
        """
        Return the support target that yields the greatest increase in own-option final strength, or None if no beneficial edge exists.
        """
        best_support_target = None
        best_support_gain = 0.0
        for target in support_targets:
            candidate_edge = (source_agent_id, target)
            if candidate_edge in supps:
                continue
            baseline_strength = self._option_strength_with_relations(own_option, atts, supps, qbaf_args, strength_cache)
            candidate_strength = self._option_strength_with_relations(own_option, atts, supps + [candidate_edge], qbaf_args, strength_cache)
            support_gain = candidate_strength - baseline_strength
            if support_gain > best_support_gain:
                best_support_gain = support_gain
                best_support_target = target
        return best_support_target


    def _best_attack_target(self, source_agent_id: str, attack_targets: list[str], mapping: dict[str, str],
                             atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                             qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> str | None:
        """
        Return the attack target that causes the greatest reduction in the opponent option's final strength, or None if no harmful edge exists.
        """
        best_attack_target = None
        best_harm = 0.0
        for target_agent_id in attack_targets:
            opponent_option = mapping.get(target_agent_id)
            if not opponent_option:
                continue
            candidate_edge = (source_agent_id, target_agent_id)
            if candidate_edge in atts:
                continue
            baseline = self._option_strength_with_relations(opponent_option, atts, supps, qbaf_args, strength_cache)
            candidate = self._option_strength_with_relations(opponent_option, atts + [candidate_edge], supps, qbaf_args, strength_cache)
            opponent_harm = baseline - candidate
            if opponent_harm > best_harm:
                best_harm = opponent_harm
                best_attack_target = target_agent_id
        return best_attack_target


    def _option_strength_with_relations(self, option: str, atts: list[tuple[str, str]], supps: list[tuple[str, str]],
                                        qbaf_args: _QBAFArguments, strength_cache: _StrengthCache) -> float:
        """
        Evaluate and cache the final strength of an option under the given attack/support relations.
        """
        cache_key = (option, tuple(sorted(atts)), tuple(sorted(supps)))
        if cache_key in strength_cache:
            return strength_cache[cache_key]
        qbaf = QBAFramework(qbaf_args.args, qbaf_args.initial_strengths, atts, supps, semantics=self._semantics)
        final_strength = cast(float, qbaf.final_strengths.get(option, 0.0))
        strength_cache[cache_key] = final_strength
        return final_strength


    def _compute_winner_snapshot(self, qbaf: QBAFramework, options: list[str]) -> _WinnerSnapshot:
        """
        Extract final strengths from a solved QBAF and identify the top two options by strength.
        """
        final_strength_by_option = {option: cast(float, qbaf.final_strengths.get(option, 0.0)) for option in options}
        ranked = sorted(final_strength_by_option.items(), key=lambda item: item[1], reverse=True)
        winner, winner_strength = ranked[0]
        second_best_strength = ranked[1][1] if len(ranked) > 1 else winner_strength
        return _WinnerSnapshot(final_strength_by_option, winner, winner_strength, second_best_strength)




    ##################################
    # STATISTICS HELPERS
    ##################################
    def _accumulate_option_outcomes(self, stats: _Stats, snapshot: _WinnerSnapshot) -> None:
        """
        Add one permutation's option strengths, win, and margin to the running accumulators.
        """
        for option, strength in snapshot.final_strength_by_option.items():
            stats.strength_sum_by_option[option] += strength
        stats.win_count_by_option[snapshot.winner] += 1
        stats.win_margin_sum_by_option[snapshot.winner] += snapshot.winner_strength - snapshot.second_best_strength


    def _accumulate_agent_outcomes(self, stats: _Stats, agent_ids: list[str], winner: str, qbaf: QBAFramework) -> None:
        """
        Add one permutation's gradient contributions and recommendation win counts to the running accumulators.
        """
        for agent_id in agent_ids:
            influence = cast(float, determine_gradient_ctrb(winner, {agent_id}, qbaf))
            stats.influence_sum_by_agent_id[agent_id] += influence
            stats.influence_square_sum_by_agent_id[agent_id] += influence * influence
            stats.recommendation_win_count_by_agent_id[agent_id] += int(stats.recommendation_by_agent_id.get(agent_id) == winner)


    def _build_option_stats(self, stats: _Stats, options: list[str], n: float) -> dict[str, dict[str, float]]:
        """
        Compute final summary statistics for each option from the accumulated Monte Carlo data.
        """
        return {option: self._option_stat_entry(stats, option, n) for option in options}

    def _option_stat_entry(self, stats: _Stats, option: str, n: float) -> dict[str, float]:
        """
        Build the stats dict for a single option: mean strength, win rate, and mean winning margin.
        """
        wins = stats.win_count_by_option[option]
        return {
            "mean_strength_over_permutations": round(stats.strength_sum_by_option[option] / n, 4),
            "top_rank_win_rate_over_permutations": round(wins / n, 4),
            "mean_winning_margin_against_second_best": round(stats.win_margin_sum_by_option[option] / wins, 4) if wins else 0.0,
        }


    def _build_agent_stats(self, stats: _Stats, agent_ids: list[str], n: float) -> list[tuple[str, dict[str, float]]]:
        """
        Compute final summary statistics for each agent, sorted by absolute mean influence descending.
        """
        entries = [(agent_id, self._agent_stat_entry(stats, agent_id, n)) for agent_id in agent_ids]
        return sorted(entries, key=lambda pair: abs(pair[1]["mean_influence"]), reverse=True)


    def _agent_stat_entry(self, stats: _Stats, agent_id: str, n: float) -> dict[str, float]:
        """
        Build the stats dict for a single agent: mean influence, influence std, and recommendation win rate.
        """
        mean_influence = stats.influence_sum_by_agent_id[agent_id] / n
        variance = stats.influence_square_sum_by_agent_id[agent_id] / n - mean_influence ** 2
        return {
            "mean_influence": round(mean_influence, 4),
            "influence_std": round(math.sqrt(max(0.0, variance)), 4),
            "recommendation_win_rate": round(stats.recommendation_win_count_by_agent_id[agent_id] / n, 4),
        }


    def _build_arguments(self, agent_recommendations: list[tuple[Agent, str, str]]) -> _QBAFArguments:
        """
        Flatten agents and their recommended options into a unified argument list with initial strengths and agent-to-option mapping.
        """
        agent_args = [agent.id for agent in self._agents]
        args = list(dict.fromkeys(agent_args + [option for _, option, _ in agent_recommendations]))
        strengths = {agent.id: agent.strength for agent in self._agents} | {option: 0.5 for _, option, _ in agent_recommendations}
        mapping = {agent.id: option for agent, option, _ in agent_recommendations}
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
