"""sar_scoring.py – SAR-discriminative scoring from GNINA docking results.

Role in the Pipeline
--------------------
    docking → sar_scoring → refinement

This module takes docking results for a full SAR compound series (actives +
inactives) docked into a single (conformation, pocket) pair, and computes
enrichment-based discrimination metrics:

  - AUC-ROC: area under the ROC curve for active vs inactive ranking
  - Enrichment Factor at 1%, 5%, 10%: fraction of actives recovered at a
    given fraction of the ranked list

The primary signal is the best GNINA score per compound (most negative
CNN affinity or Vina score). A (conformation, pocket) pair is "good" if
actives consistently outscore inactives.

When only a single compound is provided (``--ligand-smiles`` mode), SAR
scoring is not applicable and this module is not called; ranking is done
directly by docking score in the pipeline.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt

from draco.docking import GninaDockResult, PocketDockResult


# ─────────────────────────────────────────────────────────────────────────────
# Public data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SARScoreResult:
    """SAR-discrimination metrics for one (conformation, pocket) pair."""

    frame_index: int
    """MD frame index (conformation ID)."""

    pocket_id: int
    """Pocketeer pocket ID."""

    auc_roc: float
    """AUC-ROC for active vs inactive discrimination (0.5 = random, 1.0 = perfect)."""

    auc_pr: float
    """AUC-PR for active vs inactive discrimination (average precision)."""

    enrichment_1pct: float
    """Enrichment factor at 1% of the ranked compound list."""

    enrichment_5pct: float
    """Enrichment factor at 5% of the ranked compound list."""

    enrichment_10pct: float
    """Enrichment factor at 10% of the ranked compound list."""

    n_actives: int
    n_inactives: int

    active_mean_score: float
    """Mean of best docking scores for active compounds."""

    inactive_mean_score: float
    """Mean of best docking scores for inactive compounds."""

    active_std_score: float
    """Std-dev of best docking scores for active compounds (population std)."""

    inactive_std_score: float
    """Std-dev of best docking scores for inactive compounds (population std)."""

    active_min_score: float
    """Minimum of best docking scores for actives."""

    active_max_score: float
    """Maximum of best docking scores for actives."""

    inactive_min_score: float
    """Minimum of best docking scores for inactives."""

    inactive_max_score: float
    """Maximum of best docking scores for inactives."""

    active_best_score: float
    """Best (extreme) docking score among actives, consistent with `score_key` direction."""

    inactive_best_score: float
    """Best (extreme) docking score among inactives, consistent with `score_key` direction."""

    overall_min_score: float
    """Minimum of best docking scores across both actives and inactives."""

    overall_max_score: float
    """Maximum of best docking scores across both actives and inactives."""

    mean_rank_active_minus_inactive: float
    """Mean(score) difference after mapping to a 'higher is better' rank direction."""

    active_best_scores: list[float]
    """Best docking score per active compound (most negative = best binding)."""

    inactive_best_scores: list[float]
    """Best docking score per inactive compound."""

    score_key: str
    """Which GNINA score was used: 'cnn_affinity', 'vina_score', or 'cnn_score'."""

    undocked_actives: list[str] = field(default_factory=list)
    """Active compound names that produced no docking poses (GNINA failed)."""

    undocked_inactives: list[str] = field(default_factory=list)
    """Inactive compound names that produced no docking poses."""


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def compute_sar_discrimination(
    frame_index: int,
    pocket_result: PocketDockResult,
    active_names: set[str],
    inactive_names: set[str],
    *,
    score_key: str = "cnn_affinity",
) -> SARScoreResult:
    """Compute SAR discrimination score for a (conformation, pocket) pair.

    Parameters
    ----------
    frame_index:
        Index of the MD conformation (frame) being scored.
    pocket_result:
        Docking results for all compounds in this pocket (from
        ``docking.dock_ligands_to_pocket``).
    active_names:
        Set of compound names that are actives.
    inactive_names:
        Set of compound names that are inactives.
    score_key:
        Which GNINA score to use for ranking. Options:
        - ``'cnn_affinity'`` (default) – CNN predicted affinity in **pK** units (higher = better)
        - ``'vina_score'`` – Vina minimized affinity (kcal/mol; more negative = better)
        - ``'cnn_score'`` – CNN pose quality (0–1; higher = better)

    Returns
    -------
    SARScoreResult
    """
    # Ranking direction: vina_score → lower (more negative) is better.
    # cnn_affinity (pK), cnn_vs (0-1), and cnn_score (0-1) → higher is better.
    higher_is_better_score = score_key in ("cnn_affinity", "cnn_score", "cnn_vs")

    active_scores: list[float] = []
    inactive_scores: list[float] = []
    undocked_actives: list[str] = []
    undocked_inactives: list[str] = []

    for name in active_names:
        score = _best_score(pocket_result.results.get(name, []), score_key)
        if score is None:
            undocked_actives.append(name)
        else:
            active_scores.append(score)

    for name in inactive_names:
        score = _best_score(pocket_result.results.get(name, []), score_key)
        if score is None:
            undocked_inactives.append(name)
        else:
            inactive_scores.append(score)

    all_scores = np.array(active_scores + inactive_scores, dtype=np.float64)
    labels = np.array(
        [1] * len(active_scores) + [0] * len(inactive_scores), dtype=int
    )

    n_actives = len(active_scores)
    n_inactives = len(inactive_scores)
    if n_actives < 1 or n_inactives < 1:
        warnings.warn(
            f"Frame {frame_index} pocket {pocket_result.pocket_id}: "
            f"insufficient docked compounds for discrimination metrics "
            f"({n_actives} actives, {n_inactives} inactives).",
            stacklevel=2,
        )

    active_arr = np.array(active_scores, dtype=np.float64)
    inactive_arr = np.array(inactive_scores, dtype=np.float64)

    # Map scores so that "higher is better" always holds for rank-space metrics.
    if higher_is_better_score:
        active_rank = active_arr
        inactive_rank = inactive_arr
    else:
        active_rank = -active_arr
        inactive_rank = -inactive_arr

    # Stats (safe defaults for empty lists).
    def _safe_mean(x: np.ndarray) -> float:
        return float(x.mean()) if x.size else 0.0

    def _safe_std(x: np.ndarray) -> float:
        return float(x.std(ddof=0)) if x.size else 0.0

    def _safe_min(x: np.ndarray) -> float:
        return float(x.min()) if x.size else 0.0

    def _safe_max(x: np.ndarray) -> float:
        return float(x.max()) if x.size else 0.0

    active_mean = _safe_mean(active_arr)
    inactive_mean = _safe_mean(inactive_arr)
    active_std = _safe_std(active_arr)
    inactive_std = _safe_std(inactive_arr)
    active_min = _safe_min(active_arr)
    active_max = _safe_max(active_arr)
    inactive_min = _safe_min(inactive_arr)
    inactive_max = _safe_max(inactive_arr)

    if higher_is_better_score:
        active_best = active_max
        inactive_best = inactive_max
    else:
        active_best = active_min
        inactive_best = inactive_min

    overall_min = _safe_min(all_scores)
    overall_max = _safe_max(all_scores)

    mean_rank_active_minus_inactive = _safe_mean(active_rank) - _safe_mean(inactive_rank)

    auc = _roc_auc(all_scores, labels, higher_is_better=higher_is_better_score)
    auc_pr = _auprc_average_precision(all_scores, labels, higher_is_better=higher_is_better_score)
    ef1 = _enrichment_factor(all_scores, labels, fraction=0.01, higher_is_better=higher_is_better_score)
    ef5 = _enrichment_factor(all_scores, labels, fraction=0.05, higher_is_better=higher_is_better_score)
    ef10 = _enrichment_factor(all_scores, labels, fraction=0.10, higher_is_better=higher_is_better_score)

    return SARScoreResult(
        frame_index=frame_index,
        pocket_id=pocket_result.pocket_id,
        auc_roc=auc,
        auc_pr=auc_pr,
        enrichment_1pct=ef1,
        enrichment_5pct=ef5,
        enrichment_10pct=ef10,
        n_actives=n_actives,
        n_inactives=n_inactives,
        active_mean_score=active_mean,
        inactive_mean_score=inactive_mean,
        active_std_score=active_std,
        inactive_std_score=inactive_std,
        active_min_score=active_min,
        active_max_score=active_max,
        inactive_min_score=inactive_min,
        inactive_max_score=inactive_max,
        active_best_score=active_best,
        inactive_best_score=inactive_best,
        overall_min_score=overall_min,
        overall_max_score=overall_max,
        mean_rank_active_minus_inactive=mean_rank_active_minus_inactive,
        active_best_scores=active_scores,
        inactive_best_scores=inactive_scores,
        score_key=score_key,
        undocked_actives=undocked_actives,
        undocked_inactives=undocked_inactives,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric implementations (pure numpy, no sklearn required)
# ─────────────────────────────────────────────────────────────────────────────

def _roc_auc(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    *,
    higher_is_better: bool = False,
) -> float:
    """Compute AUC-ROC without sklearn.

    Parameters
    ----------
    scores:
        Score for each compound (shape N,).
    labels:
        Binary labels: 1 = active, 0 = inactive (shape N,).
    higher_is_better:
        If True, compounds with higher scores are predicted as actives.
        If False (default), lower (more negative) scores are predicted as actives.
    """
    n = len(scores)
    if n == 0:
        return 0.5

    # Sort descending so the "best" compound is first.
    # For affinity modes (lower is better): sort ascending, then reverse → most negative first.
    sort_idx = np.argsort(scores)
    if higher_is_better:
        sort_idx = sort_idx[::-1]  # highest score first
    # else: lowest score first (most negative affinity = best binder)

    sorted_labels = labels[sort_idx]

    n_pos = sorted_labels.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Trapezoid AUC from sorted labels
    tpr = np.cumsum(sorted_labels) / n_pos
    fpr = np.cumsum(1 - sorted_labels) / n_neg
    # Prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # NumPy 2.0 removed `np.trapz` in favor of `np.trapezoid`.
    # Keep compatibility across NumPy versions.
    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:  # pragma: no cover
        auc = float(np.trapz(tpr, fpr))
    return auc


def _enrichment_factor(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    *,
    fraction: float,
    higher_is_better: bool = False,
) -> float:
    """Enrichment factor at ``fraction`` of the ranked list.

    EF_x = (actives_in_top_x%) / (expected_actives_in_top_x%)
         = (hits_found / hits_expected)
    where hits_expected = fraction * n_actives_total.
    """
    n = len(scores)
    if n == 0:
        return 0.0

    sort_idx = np.argsort(scores)
    if higher_is_better:
        sort_idx = sort_idx[::-1]

    n_top = max(1, int(np.ceil(fraction * n)))
    top_labels = labels[sort_idx[:n_top]]
    n_actives_total = labels.sum()
    if n_actives_total == 0:
        return 0.0

    hits_found = top_labels.sum()
    hits_expected = fraction * n_actives_total
    return float(hits_found / hits_expected) if hits_expected > 0 else 0.0


def _auprc_average_precision(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    *,
    higher_is_better: bool = False,
) -> float:
    """Compute AUC-PR as *average precision* (AP) without sklearn.

    Uses the standard AP definition for ranked binary labels:
      AP = mean(precision_at_each_positive_position)
    """
    n = len(scores)
    if n == 0:
        return 0.0

    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0

    sort_idx = np.argsort(scores)
    if higher_is_better:
        sort_idx = sort_idx[::-1]
    sorted_labels = labels[sort_idx]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    precision = tp / (tp + fp)

    pos_idx = np.where(sorted_labels == 1)[0]
    # Recall increases by 1/n_pos at each positive, so AP is just the mean precision
    # at positive indices.
    return float(precision[pos_idx].mean()) if pos_idx.size else 0.0


def _best_score(
    poses: list[GninaDockResult],
    score_key: str,
) -> float | None:
    """Return the best (extreme) score from a list of docked poses.

    Returns None if ``poses`` is empty (compound not docked).
    For ``cnn_affinity`` (pK): returns maximum. For ``vina_score``: minimum (most negative).
    For ``cnn_score`` (0–1) and ``cnn_vs`` (0-1): returns maximum.
    """
    if not poses:
        return None
    vals = [getattr(p, score_key) for p in poses]
    if score_key in ("cnn_score", "cnn_affinity", "cnn_vs"):
        return max(vals)
    return min(vals)  # vina: most negative = best
