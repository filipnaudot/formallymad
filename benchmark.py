import random
from concurrent.futures import ThreadPoolExecutor

from data import SymptomDataset
from formallymad.agent import Agent
import formallymad.prompts as PROMPTS
from formallymad.qbaf import QBAFResolver, normalize_attribution_strengths
from formallymad.ui import FormallyMADUI


def _apply_attribution_strengths(workers: list[Agent], attribution_scores: dict[str, float]) -> dict[str, float]:
    """
    Normalize raw llmSHAP attribution scores to [0, 1] and update each worker's strength in place.

    :param workers: List of worker agents whose strengths will be updated.
    :param attribution_scores: Raw Shapley scores keyed by agent id.
    :return: Normalized scores keyed by agent id.
    """
    normalized = normalize_attribution_strengths(attribution_scores)
    for worker in workers:
        if worker.id in normalized:
            worker.update_strength(normalized[worker.id])
    return normalized


def _build_query(samples: list[SymptomDataset]) -> tuple[SymptomDataset, list[str], str]:
    """
    Draw a random sample from the dataset and build a multiple-choice diagnostic query.

    :param samples: Full list of loaded symptom dataset entries.
    :return: Tuple of (selected sample, shuffled options list, formatted query string).
    """
    sample = random.choice(samples)
    distractors = random.sample([s.disease_name for s in samples if s.disease_name != sample.disease_name], k=3)
    options = random.sample([sample.disease_name] + distractors, k=4)
    options_text = "\n".join(f"  {chr(65 + i)}. {name}" for i, name in enumerate(options))
    query = f"""A patient presents with the following symptoms: {', '.join(sample.symptom_list)}.
                What disease does this patient most likely have?
                Choose one of the following options: {options_text}
                Your recommendation must be the exact disease name as written above, verbatim.
            """
    return sample, options, query


def _normalize_to_option(recommendation: str, options: list[str]) -> str:
    """
    Map a free-form model recommendation to the closest option string.
    Matches longest option first to avoid partial overlaps. Returns the original string if no match is found.

    :param recommendation: Raw recommendation string from an agent.
    :param options: List of valid option strings to match against.
    :return: The matched option string, or the original recommendation if no match is found.
    """
    cleaned = recommendation.strip().rstrip(".,;").lower()
    for option in sorted(options, key=len, reverse=True):
        if option.lower() == cleaned or option.lower() in cleaned:
            return option
    return recommendation


def main() -> None:
    ui = FormallyMADUI()
    ui.banner(text="Formally MAD - Benchmark")

    workers = [
        Agent(id="Deceiver", strength=0.3, role="You are a deceiver. Always try to convince the oracle of the wrong recommendation."),
        Agent(id="A2", strength=0.5),
        Agent(id="A3", strength=0.2),
        Agent(id="A4", strength=0.2),
        Agent(id="A5", strength=0.2),
        Agent(id="A6", strength=0.2),
    ]
    oracle = Agent(id="oracle", system_prompt=PROMPTS.ORACLE_PROMPT)
    qbaf = QBAFResolver(workers, monte_carlo_permutations=10, semantics_aware=True, visualize=True)
    samples = SymptomDataset.load()

    while True:
        try:
            ui.console.input("[grey62]Press Enter for a new sample (Ctrl-C to quit)...[/grey62]")
        except (KeyboardInterrupt, EOFError):
            break

        sample, options, query = _build_query(samples)

        with ui.loading("Collecting recommendations..."):
            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                futures = [(agent, pool.submit(agent.recommend, query)) for agent in workers]
                recommendations = [(agent, future.result()) for agent, future in futures]

        cleaned_recommendations = [(agent, _normalize_to_option(rec.recommendation, options), rec.motivation) for agent, rec in recommendations]
        ui.show_proposals((agent.id, recommendation, motivation) for agent, recommendation, motivation in cleaned_recommendations)

        with ui.loading("Computing llmSHAP attribution..."):
            final, attribution_scores = oracle.synthesize_with_attribution(query, recommendations)

        normalized_strengths = _apply_attribution_strengths(workers, attribution_scores)
        winner, _ = qbaf.resolve(cleaned_recommendations)
        ui.show_agent_metrics(qbaf.last_agent_stats, strength_by_agent_id=normalized_strengths)
        ui.show_result("QBAF winner", winner)
        ui.show_assistant(final)
        ui.show_result("Ground truth", sample.disease_name, color="green")


if __name__ == "__main__":
    main()
