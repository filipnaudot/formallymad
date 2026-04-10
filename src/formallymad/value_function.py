from llmSHAP.generation import Generation
from llmSHAP.value_functions import TFIDFCosineSimilarity, ValueFunction


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
        Used for label extraction via substring matching (longest-first).
    label_weight:
        Weight in [0, 1] controlling how strongly a label match/mismatch shifts the score.
        Higher values make label changes dominate over TF-IDF token similarity.
        Default: 0.5 (equal split between label signal and TF-IDF signal).
    """

    def __init__(self, options: list[str], label_weight: float = 0.5) -> None:
        self._options = sorted([option.lower() for option in options], key=len, reverse=True)
        self._tfidf = TFIDFCosineSimilarity()
        self._label_weight = label_weight

    def _extract_label(self, text: str) -> str | None:
        """Return the first (longest) option found in text, or None if no match."""
        text_lower = text.lower()
        for option in self._options:
            if option in text_lower:
                return option
        return None

    def __call__(self, base_generation: Generation, coalition_generation: Generation) -> float:
        """
        Score the coalition generation relative to the base generation.

        Same-label pairs → score in [label_weight, 1.0].
        Different-label pairs → score in [0.0, 1 - label_weight].
        Falls back to raw TF-IDF when label extraction fails for either output.
        """
        tfidf_sim = self._tfidf(base_generation, coalition_generation)
        base_label = self._extract_label(base_generation.output)
        coalition_label = self._extract_label(coalition_generation.output)

        if base_label is None or coalition_label is None:
            return tfidf_sim
        if base_label == coalition_label:
            return self._label_weight + (1.0 - self._label_weight) * tfidf_sim
        else:
            return (1.0 - self._label_weight) * tfidf_sim
