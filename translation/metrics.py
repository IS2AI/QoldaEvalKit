"""Translation metrics: XCOMET as the headline, BLEU and chrF++ alongside.

XCOMET is a neural metric run locally on a GPU, not an API judge — the model is
loaded once per run and every direction is scored through the same instance,
because loading a 10.7B checkpoint six times would dominate the run.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("qoldaevalkit")

DEFAULT_MODEL = "Unbabel/XCOMET-XXL"


class XCometScorer:
    """Lazily-loaded XCOMET scorer, reused across directions."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 8,
                 gpus: int = 1, cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from comet import download_model, load_from_checkpoint
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "XCOMET scoring needs the comet package: pip install unbabel-comet"
            ) from exc

        logger.info("Loading %s (this downloads several GB the first time)",
                    self.model_name)
        print(f"  [xcomet] loading {self.model_name} …", flush=True)
        try:
            path = download_model(self.model_name,
                                  saving_directory=self.cache_dir)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not fetch {self.model_name}. It is a gated model: accept "
                f"the licence at https://huggingface.co/{self.model_name} with the "
                f"account behind HF_TOKEN, then retry. Original error: {exc}"
            ) from exc

        self._model = load_from_checkpoint(path)
        self._model.eval()
        return self._model

    def score(self, triplets: Sequence[Dict[str, str]]
              ) -> Tuple[List[float], Optional[float]]:
        """Per-segment scores and the system score, both in 0-1."""
        if not triplets:
            return [], None
        model = self._load()
        output = model.predict(list(triplets), batch_size=self.batch_size,
                               gpus=self.gpus, progress_bar=True)
        scores = [float(s) for s in output.scores]
        system = float(output.system_score) if output.system_score is not None else None
        return scores, system


# ---------------------------------------------------------------------------
# Surface metrics — free, and a useful sanity check on the neural score
# ---------------------------------------------------------------------------

def surface_metrics(hypotheses: Sequence[str], references: Sequence[str],
                    target_lang: str) -> Dict[str, float]:
    """Corpus BLEU and chrF++ , or an empty dict when sacrebleu is absent."""
    if not hypotheses:
        return {}
    try:
        import sacrebleu
    except ImportError:
        logger.info("sacrebleu not installed; reporting XCOMET only")
        return {}

    refs = [list(references)]
    # Kazakh and Russian are not whitespace-pathological, but chrF++ is the
    # more reliable of the two for morphologically rich targets.
    bleu = sacrebleu.corpus_bleu(list(hypotheses), refs)
    chrf = sacrebleu.corpus_chrf(list(hypotheses), refs, word_order=2)
    # sacrebleu reports 0-100; stored as a 0-1 fraction so every translation
    # metric shares one scale with XCOMET and is reported as 0.3144.
    return {"bleu": round(bleu.score / 100.0, 6),
            "chrf2": round(chrf.score / 100.0, 6)}
