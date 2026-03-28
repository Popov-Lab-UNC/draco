"""constants.py – Codified parameters for the Draco pipeline."""

# ── Protein Preparation & MD Settings ───────────────────────────────────────
DEFAULT_PH = 7.4
DEFAULT_FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")
DEFAULT_WATER_MODEL = "tip3pfb"

# Ligand openmm force fields
DEFAULT_OPENFF_FORCEFIELD = "openff-2.3.0"

# OpenMM / MD Physics Constants
KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2: float = 418.4

# Ion / water residue names excluded from protein
SOLVENT_ION_RESNAMES: set[str] = {
    "HOH", "WAT", "TIP3", "SPC", "T3P", "T4P", "T5P",
    "NA", "CL", "Na+", "Cl-", "NA+", "CL-",
    "K", "K+", "MG", "CA", "ZN", "FE",
}

# Modeller solver model aliases
MODELLER_SOLVENT_MODEL_ALIASES: dict[str, str] = {"tip3pfb": "tip3p"}

# ── Docking / GNINA ─────────────────────────────────────────────────────────
DEFAULT_GNINA_BINARY = "gnina"
DEFAULT_GNINA_TIMEOUT_SECONDS = 300
DEFAULT_DOCKING_PADDING_ANGSTROM = 4.0
DEFAULT_DOCKING_MIN_SIZE_ANGSTROM = 15.0

# ── Refinement ──────────────────────────────────────────────────────────────
DEFAULT_REFINEMENT_SHELL_RADIUS = 8.0
DEFAULT_REFINEMENT_PROTEIN_RESTRAINT_K = 10.0
DEFAULT_REFINEMENT_MAX_ITERATIONS = 500
