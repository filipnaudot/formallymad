import random
from types import SimpleNamespace

from formallymad.qbaf import QBAFResolver


for num_agents in [5, 10, 20, 40]:
    changed_winner = 0
    for _ in range(100):
        agents = [SimpleNamespace(id=f"A{i}", strength=random.random()) for i in range(num_agents)]
        recommendations = [(agent, f"R{random.randrange(5)}", "") for agent in agents]
        resolver = QBAFResolver(agents, semantics_aware=True, monte_carlo_permutations=len(agents), seed=42) # type: ignore
        winner, _ = resolver.resolve(recommendations)  # type: ignore[arg-type]
        changed_winner += resolver.last_option_stats[winner]["top_rank_win_rate_over_permutations"] < 1.0
    print(f"{num_agents} agents: {changed_winner}/100 changed winner across permutations")
