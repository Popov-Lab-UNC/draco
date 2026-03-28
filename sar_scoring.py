"""sar_scoring.py – SAR-discriminative scoring from GNINA docking results.

Role in the Pipeline
--------------------
    gnina_docking → sar_scoring → final_refinement

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

from gnina_docking import GninaDockResult, PocketDockResult


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

    enrichment_1pct: float
    """Enrichment factor at 1% of the ranked compound list."""

    enrichment_5pct: float
    """Enrichment factor at 5% of the ranked compound list."""

    enrichment_10pct: float
    """Enrichment factor at 10% of the ranked compound list."""

    n_actives: int
    n_inactives: int

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
        ``gnina_docking.dock_ligands_to_pocket``).
    active_names:
        Set of compound names that are actives.
    inactive_names:
        Set of compound names that are inactives.
    score_key:
        Which GNINA score to use for ranking. Options:
        - ``'cnn_affinity'`` (default) – CNN predicted binding affinity (kcal/mol)
        - ``'vina_score'`` – Vina minimized affinity (kcal/mol)
        - ``'cnn_score'`` – CNN pose quality (0–1; higher = better, opposite sign)

    Returns
    -------
    SARScoreResult
    """
    # Determine sign convention: for affinity/energy scores lower (more negative)
    # is better. For cnn_score higher is better. We normalize so that "more
    # positive = worse" for ranking, meaning we store raw scores and flip for AUC.
    affinity_mode = score_key in ("cnn_affinity", "vina_score")

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

    if len(active_scores) < 2 or len(inactive_scores) < 1:
        warnings.warn(
            f"Frame {frame_index} pocket {pocket_result.pocket_id}: "
            f"insufficient docked compounds for AUC computation "
            f"({len(active_scores)} actives, {len(inactive_scores)} inactives). "
            "AUC set to 0.5.",
            stacklevel=2,
        )
        return SARScoreResult(
            frame_index=frame_index,
            pocket_id=pocket_result.pocket_id,
            auc_roc=0.5,
            enrichment_1pct=0.0,
            enrichment_5pct=0.0,
            enrichment_10pct=0.0,
            n_actives=len(active_scores),
            n_inactives=len(inactive_scores),
            active_best_scores=active_scores,
            inactive_best_scores=inactive_scores,
            score_key=score_key,
            undocked_actives=undocked_actives,
            undocked_inactives=undocked_inactives,
        )

    all_scores = np.array(active_scores + inactive_scores, dtype=np.float64)
    labels = np.array(
        [1] * len(active_scores) + [0] * len(inactive_scores), dtype=int
    )

    auc = _roc_auc(all_scores, labels, higher_is_better=not affinity_mode)
    ef1 = _enrichment_factor(all_scores, labels, fraction=0.01, higher_is_better=not affinity_mode)
    ef5 = _enrichment_factor(all_scores, labels, fraction=0.05, higher_is_better=not affinity_mode)
    ef10 = _enrichment_factor(all_scores, labels, fraction=0.10, higher_is_better=not affinity_mode)

    return SARScoreResult(
        frame_index=frame_index,
        pocket_id=pocket_result.pocket_id,
        auc_roc=auc,
        enrichment_1pct=ef1,
        enrichment_5pct=ef5,
        enrichment_10pct=ef10,
        n_actives=len(active_scores),
        n_inactives=len(inactive_scores),
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


def _best_score(
    poses: list[GninaDockResult],
    score_key: str,
) -> float | None:
    """Return the best (extreme) score from a list of docked poses.

    Returns None if ``poses`` is empty (compound not docked).
    For affinity-like scores (cnn_affinity, vina_score): returns minimum (most negative).
    For cnn_score (0-1, higher=better): returns maximum.
    """
    if not poses:
        return None
    vals = [getattr(p, score_key) for p in poses]
    if score_key == "cnn_score":
        return max(vals)
    return min(vals)  # most negative = best affinity
