from __future__ import annotations

import numpy as np

from retina_fl.metrics import multilabel_metrics


def test_multilabel_metrics_use_configured_threshold_and_exact_match():
    targets = np.array([[1, 0, 1], [0, 1, 0]])
    probabilities = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]])
    metrics = multilabel_metrics(targets, probabilities, ["normal", "glaucoma", "cataract"], 0.55)
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
