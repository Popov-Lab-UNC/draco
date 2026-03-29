"""constants.py – Centralised physical and chemical constants for Draco.

Consolidates values previously duplicated across dynamics.py,
local_minimization.py, ligand_preparation.py, and pocket_coloring.py.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Unit conversions
# ─────────────────────────────────────────────────────────────────────────────

KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2: float = 418.4
"""Convert a force constant from kcal/mol/Å² to kJ/mol/nm²."""

# ─────────────────────────────────────────────────────────────────────────────
# Force-field defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FORCEFIELD: tuple[str, ...] = ("amber14-all.xml", "amber14/tip3pfb.xml")
"""Default OpenMM ForceField XML files (AMBER14 protein + TIP3P-FB water)."""

DEFAULT_WATER_MODEL: str = "tip3pfb"
"""Chemistry label for the water model."""

# ─────────────────────────────────────────────────────────────────────────────
# Residue-name sets
# ─────────────────────────────────────────────────────────────────────────────

SOLVENT_ION_RESNAMES: frozenset[str] = frozenset({
    "HOH", "WAT", "TIP3", "SPC", "T3P", "T4P", "T5P",
    "NA", "CL", "Na+", "Cl-", "NA+", "CL-",
    "K", "K+", "MG", "CA", "ZN", "FE",
})
"""Ion / water residue names excluded when identifying protein atoms."""

# ─────────────────────────────────────────────────────────────────────────────
# Ligand pharmacophore feature labels
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_LABEL_MAP: dict[str, str] = {
    "Donor": "donor",
    "Acceptor": "acceptor",
    "PosIonizable": "cation",
    "NegIonizable": "anion",
    "Aromatic": "ring",
    "LumpedHydrophobe": "hydrophobe",
}
"""RDKit pharmacophore feature family → Draco internal label."""

# ─────────────────────────────────────────────────────────────────────────────
# Residue classification (retained from pocket_coloring for interpretability)
# ─────────────────────────────────────────────────────────────────────────────

AROMATIC_RESIDUES: frozenset[str] = frozenset({"PHE", "TYR", "TRP", "HIS"})
HYDROPHOBIC_RESIDUES: frozenset[str] = frozenset({
    "ALA", "VAL", "ILE", "LEU", "MET", "PRO",
})
POSITIVE_RESIDUES: frozenset[str] = frozenset({"ARG", "LYS"})
NEGATIVE_RESIDUES: frozenset[str] = frozenset({"ASP", "GLU"})
POLAR_RESIDUES: frozenset[str] = frozenset({"SER", "THR", "ASN", "GLN", "CYS"})
