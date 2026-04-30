from dataclasses import dataclass

from llmSHAP import PromptCodec
from llmSHAP.data_handler import DataHandler
from llmSHAP.generation import Generation
from llmSHAP.types import IndexSelection
from llmSHAP.value_functions import TFIDFCosineSimilarity, ValueFunction



@dataclass
class RecommendationGeneration(Generation):
    recommendation: str
    motivation: str


class RecommendationPromptCodec(PromptCodec):
    def __init__(self, system: str = "") -> None:
        self.system = system

    def build_prompt(self, data_handler: DataHandler, indexes: IndexSelection) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": data_handler.to_string(indexes)}]

    def parse_generation(self, model_output: object) -> RecommendationGeneration:
        recommendation = getattr(model_output, "recommendation", "")
        motivation = getattr(model_output, "motivation", "")
        return RecommendationGeneration(output=f"RECOMMENDATION: {recommendation}\nMOTIVATION: {motivation}",
                                        recommendation=recommendation,
                                        motivation=motivation)


class LabelWeightedSimilarity(ValueFunction):
    """
    Value function combining TF-IDF cosine similarity with a label-change weight.

    When the coalition output carries the same recommendation label as the base (grand-coalition)
    output, the score is boosted into [label_weight, 1.0].
    When the label differs, the score is suppressed into [0.0, 1 - label_weight].

    This addresses the all-zero Shapley issue that arises when oracle outputs share
    so many tokens that TF-IDF cannot distinguish coalition-level differences.

    Parameters
    ----------
    options:
        The valid recommendation option strings (verbatim, as presented to agents).
        Used for recommendation extraction via substring matching (longest-first).
    label_weight:
        Weight in [0, 1] controlling how strongly a label match/mismatch shifts the score.
        Higher values make label changes dominate over TF-IDF token similarity.
        Default: 0.5 (equal split between label signal and TF-IDF signal).
    """

    def __init__(self, options: list[str], label_weight: float = 0.5) -> None:
        self._options = sorted([option.lower() for option in options], key=len, reverse=True)
        self._tfidf = TFIDFCosineSimilarity()
        self._label_weight = label_weight

    def _extract_recommendation(self, recommendation: str) -> str | None:
        """Return the first (longest) option found in the recommendation field, or None."""
        text_lower = recommendation.lower()
        for option in self._options:
            if option in text_lower:
                return option
        print("NO OPTION")
        return None

    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        """
        Score the coalition generation relative to the base generation.

        Same-label pairs → score in [label_weight, 1.0].
        Different-label pairs → score in [0.0, 1 - label_weight].
        Falls back to raw TF-IDF when recommendation extraction fails for either output.
        """
        tfidf_sim = self._tfidf(base_generation, coalition_generation)
        base_recommendation = getattr(base_generation, "recommendation", "")
        coalition_recommendation = getattr(coalition_generation, "recommendation", "")
        base_label = self._extract_recommendation(base_recommendation)
        coalition_label = self._extract_recommendation(coalition_recommendation)

        if base_label is None or coalition_label is None:
            return tfidf_sim
        if base_label == coalition_label:
            return self._label_weight + (1.0 - self._label_weight) * tfidf_sim
        else:
            return (1.0 - self._label_weight) * tfidf_sim
