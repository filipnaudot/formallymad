from dataclasses import dataclass
import pytest

from formallymad import QBAFResolver



@dataclass
class FakeAgent:
    agent_id: str
    agent_strength: float
    def id(self) -> str: return self.agent_id
    def strength(self) -> float: return self.agent_strength


def test_resolve_rejects_empty_tool_proposals() -> None:
    resolver = QBAFResolver(agents=[])
    with pytest.raises(ValueError, match="tool_proposals must not be empty"): resolver.resolve([])


def test_resolve_single_tool_proposal_has_expected_basic_stats() -> None:
    agent = FakeAgent(agent_id="agent-1", agent_strength=0.8)
    resolver = QBAFResolver(agents=[agent], monte_carlo_permutations=5, seed=42)
    winner_tool_name, agent_influence = resolver.resolve([(agent, "tool-a", "use tool-a")])
    assert winner_tool_name == "tool-a"
    assert len(agent_influence) == 1
    assert agent_influence[0][0] == "agent-1"
    assert resolver.last_tool_stats["tool-a"]["top_rank_win_rate_over_permutations"] == 1.0


def test_resolve_is_reproducible_with_same_seed() -> None:
    agents = [FakeAgent(agent_id="agent-1", agent_strength=0.9),
              FakeAgent(agent_id="agent-2", agent_strength=0.6),
              FakeAgent(agent_id="agent-3", agent_strength=0.3)]
    tool_proposals = [(agents[0], "tool-a", "proposal a"),
                      (agents[1], "tool-b", "proposal b"),
                      (agents[2], "tool-a", "proposal c")]
    resolver_a = QBAFResolver(agents=agents, monte_carlo_permutations=20, seed=7)
    resolver_b = QBAFResolver(agents=agents, monte_carlo_permutations=20, seed=7)
    result_a = resolver_a.resolve(tool_proposals)
    result_b = resolver_b.resolve(tool_proposals)
    assert result_a == result_b
    assert resolver_a.last_tool_stats == resolver_b.last_tool_stats
    assert resolver_a.last_agent_stats == resolver_b.last_agent_stats
