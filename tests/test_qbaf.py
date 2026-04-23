from dataclasses import dataclass
import pytest

from formallymad import QBAFResolver
import formallymad.qbaf as qbaf_module



@dataclass
class FakeAgent:
    agent_id: str
    agent_strength: float

    @property
    def id(self) -> str: return self.agent_id

    @property
    def strength(self) -> float: return self.agent_strength


def test_resolve_rejects_empty_agent_recommendations() -> None:
    resolver = QBAFResolver(agents=[])
    with pytest.raises(ValueError, match="agent_recommendations must not be empty"): resolver.resolve([])


def test_resolve_single_option_recommendation_has_expected_basic_stats() -> None:
    agent = FakeAgent(agent_id="agent-1", agent_strength=0.8)
    resolver = QBAFResolver(agents=[agent], monte_carlo_permutations=5, seed=42) # type: ignore
    winner_option, agent_influence = resolver.resolve([(agent, "recommendation-a", "use recommendation-a")]) # type: ignore
    assert winner_option == "recommendation-a"
    assert len(agent_influence) == 1
    assert agent_influence[0][0] == "agent-1"
    assert resolver.last_option_stats["recommendation-a"]["top_rank_win_rate_over_permutations"] == 1.0


def test_resolve_is_reproducible_with_same_seed() -> None:
    agents = [FakeAgent(agent_id="agent-1", agent_strength=0.9),
              FakeAgent(agent_id="agent-2", agent_strength=0.6),
              FakeAgent(agent_id="agent-3", agent_strength=0.3)]
    agent_recommendations = [(agents[0], "recommendation-a", "motivation a"),
                             (agents[1], "recommendation-b", "motivation b"),
                             (agents[2], "recommendation-a", "motivation c")]
    resolver_a = QBAFResolver(agents=agents, monte_carlo_permutations=20, seed=7) # type: ignore
    resolver_b = QBAFResolver(agents=agents, monte_carlo_permutations=20, seed=7) # type: ignore
    result_a = resolver_a.resolve(agent_recommendations) # type: ignore
    result_b = resolver_b.resolve(agent_recommendations) # type: ignore
    assert result_a == result_b
    assert resolver_a.last_option_stats == resolver_b.last_option_stats
    assert resolver_a.last_agent_stats == resolver_b.last_agent_stats
