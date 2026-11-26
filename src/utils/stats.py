"""Paired significance tests for comparing two variants on the same test sentences.

Each variant has S runs (seeds) on the same n test sentences. The runs of one variant are not independent
samples of the test set: they share the sentences. Pooling the S x n pairs into one McNemar test therefore
overstates the sample size. The tests here keep the sentence as the unit:

* clustered_bootstrap: per sentence i, s_i = mean over seeds of 1(correct); d_i = s_i(B) - s_i(A).
  The n values d_i are independent across sentences. The 95% CI of mean(d) comes from resampling the
  sentences with replacement (the seeds of a sentence stay together), and the two-sided p-value from a
  sign-flip permutation test of mean(d) = 0.
* exact_mcnemar on the probability-averaged ensemble of each variant: n independent pairs.
* pooled McNemar over S x n pairs is kept only as a reference number.
"""
import numpy as np
from scipy.stats import binomtest


def exact_mcnemar(right_a, right_b):
    """Exact two-sided McNemar test on paired correctness vectors (bool arrays of equal length)."""
    right_a, right_b = np.asarray(right_a, bool), np.asarray(right_b, bool)
    if right_a.shape != right_b.shape:
        raise ValueError("paired vectors must have the same shape")
    only_a = int((right_a & ~right_b).sum())
    only_b = int((~right_a & right_b).sum())
    p = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"only_a": only_a, "only_b": only_b, "n_pairs": int(right_a.size), "p_value": float(p)}


def clustered_bootstrap(correct_a, correct_b, n_boot=10_000, seed=42, chunk=1_000):
    """Bootstrap over sentences of the difference in mean accuracy (B - A).

    correct_a: (n, S_a) 0/1 matrix, correct_b: (n, S_b), rows aligned on the same sentences.
    Returns mean_diff, ci_95_low, ci_95_high (accuracy fractions), p_value_two_sided (sign-flip
    permutation of d_i), p_bootstrap_two_sided (share of bootstrap means on the other side of 0, x2).
    """
    a = np.asarray(correct_a, float)
    b = np.asarray(correct_b, float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        raise ValueError("both matrices need the same, non-zero number of sentences")
    d = b.mean(1) - a.mean(1)
    n = d.size
    observed = float(d.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    flips = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        boot[start:start + size] = d[rng.integers(0, n, (size, n))].mean(1)
        signs = rng.integers(0, 2, (size, n)) * 2 - 1
        flips[start:start + size] = (signs * d).mean(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    p_perm = (1 + int((np.abs(flips) >= abs(observed) - 1e-12).sum())) / (n_boot + 1)
    if observed == 0 and np.all(d == 0):
        p_perm = 1.0
    p_boot = min(1.0, 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
    return {"mean_diff": observed, "ci_95_low": float(low), "ci_95_high": float(high),
            "p_value_two_sided": float(min(1.0, p_perm)), "p_bootstrap_two_sided": p_boot,
            "n_sentences": int(n), "n_seeds_a": int(a.shape[1]), "n_seeds_b": int(b.shape[1]),
            "n_boot": int(n_boot)}


def ensemble_predictions(probabilities):
    """Argmax of the mean class probabilities over runs; probabilities: (S, n, C)."""
    return np.asarray(probabilities, float).mean(0).argmax(-1)


def pairwise_agreement(predictions):
    """Mean share of sentences on which two runs predict the same label; predictions: (R, n)."""
    predictions = np.asarray(predictions)
    runs = predictions.shape[0]
    if runs < 2:
        return float("nan")
    values = [(predictions[i] == predictions[j]).mean() for i in range(runs) for j in range(i + 1, runs)]
    return float(np.mean(values))


def majority_vote(predictions, probabilities=None, num_classes=3):
    """Most frequent label per sentence; ties go to the label with the larger summed probability
    (or the smaller label index without probabilities). predictions: (R, n), probabilities: (R, n, C)."""
    predictions = np.asarray(predictions)
    votes = np.stack([(predictions == c).sum(0) for c in range(num_classes)], -1).astype(float)
    if probabilities is not None:
        votes += 1e-6 * np.asarray(probabilities, float).sum(0) / max(1, predictions.shape[0])
    return votes.argmax(-1)


def recall_per_class(gold, pred, num_classes=3):
    gold, pred = np.asarray(gold), np.asarray(pred)
    return [float((pred[gold == c] == c).mean()) if (gold == c).any() else float("nan") for c in range(num_classes)]


def collapsed_classes(pred, num_classes=3, min_share=0.05):
    """Classes predicted for fewer than min_share of the sentences (a collapsed class)."""
    pred = np.asarray(pred)
    return [c for c in range(num_classes) if (pred == c).mean() < min_share]
