"""Small-MLP learned calibrator.

An MLP calibrator strictly dominates Platt/temperature in expressivity and
strictly dominates isotonic in smoothness. It matters because some
retrieval signals (e.g. cross-encoder logits on BEIR SciFact) show
*bimodal* calibration curves that neither a parametric sigmoid nor a
step-function isotonic fit captures well. We use a two-layer MLP with
``hidden=16`` and sigmoid output, trained by L-BFGS on binary cross
entropy (switching to scikit-learn's ``MLPClassifier`` keeps the
dependency set minimal).

We regularise with weak weight decay (``alpha=1e-4``). On small
calibration partitions the MLP is prone to overfitting; the benchmark
reports calibration metrics on a completely disjoint test split so
overfitting is always visible in the numbers.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCalibrator


class LearnedCalibrator(BaseCalibrator):
    name = "learned_mlp"

    def __init__(self, hidden: int = 16, alpha: float = 1e-4, max_iter: int = 500, random_state: int = 0) -> None:
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._model = None
        self._platt_fallback = None
        self._fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "LearnedCalibrator":
        s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if s.size == 0 or np.all(y == y[0]):
            # Trivially calibrate to the prior.
            from .platt import PlattCalibrator

            self._platt_fallback = PlattCalibrator().fit(s.reshape(-1), y)
            self._fitted = True
            return self

        try:
            from sklearn.neural_network import MLPClassifier  # type: ignore

            mlp = MLPClassifier(
                hidden_layer_sizes=(self.hidden,),
                activation="tanh",
                solver="lbfgs",
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            mlp.fit(s, y)
            self._model = mlp
        except Exception:  # pragma: no cover
            from .platt import PlattCalibrator

            self._platt_fallback = PlattCalibrator().fit(s.reshape(-1), y)
        self._fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LearnedCalibrator.fit must be called before transform")
        s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        if self._model is not None:
            p = self._model.predict_proba(s)[:, 1]
            return np.clip(p, 1e-6, 1 - 1e-6)
        assert self._platt_fallback is not None
        return self._platt_fallback.transform(s.reshape(-1))
