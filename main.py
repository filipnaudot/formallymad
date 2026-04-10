from concurrent.futures import ThreadPoolExecutor

from formallymad.agent import Agent
from formallymad.prompts import ORACLE_PROMPT
from formallymad.qbaf import QBAFResolver
from formallymad.ui import FormallyMADUI


def main() -> None:
    ui = FormallyMADUI()
    ui.banner(text="Formally MAD")

    workers = [
        Agent(id="Deceiver", strength=0.3, role="You are a deceiver. Always try to convince the oracle of the wrong recommendation."),
        Agent(id="A2", strength=0.5),
        Agent(id="A3", strength=0.2),
        Agent(id="A4", strength=0.2),
        Agent(id="A5", strength=0.2),
        Agent(id="A6", strength=0.2),
    ]

    oracle = Agent(id="oracle", system_prompt=ORACLE_PROMPT)
    qbaf = QBAFResolver(workers, monte_carlo_permutations=10, semantics_aware=True, visualize=True)

    while True:
        try:
            query = ui.ask()
        except (KeyboardInterrupt, EOFError):
            break

        with ui.loading("Collecting recommendations..."):
            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                futures = [(agent, pool.submit(agent.recommend, query)) for agent in workers]
                recommendations = [(agent, future.result()) for agent, future in futures]

        winner, _ = qbaf.resolve([(a, rec.recommendation, rec.motivation) for a, rec in recommendations])
        ui.show_proposals((a.id, rec.recommendation, rec.motivation) for a, rec in recommendations)
        ui.show_agent_metrics(qbaf.last_agent_stats)
        ui.show_result("QBAF winner", winner)

        with ui.loading("Oracle is synthesizing..."):
            final = oracle.synthesize(query, recommendations)
        ui.show_assistant(final)


if __name__ == "__main__":
    main()
