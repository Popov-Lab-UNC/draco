from __future__ import annotations

from dataclasses import dataclass
import io
import hashlib
from pathlib import Path

try:
    from openmm.app import ForceField, Modeller, PDBFile, Topology
    from openmm.unit import Quantity
except ImportError:  # pragma: no cover
    from simtk.openmm.app import ForceField, Modeller, PDBFile, Topology  # type: ignore
    from simtk.unit import Quantity  # type: ignore

try:
    from pdbfixer import PDBFixer
    _PDBFIXER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDBFIXER_AVAILABLE = False

_DEFAULT_FORCEFIELD_FILES = ("amber14-all.xml", "amber14/tip3pfb.xml")


@dataclass(frozen=True)
class PreparedProtein:
    """Protein topology and positions after PDBFixer repair and protonation."""

    topology: Topology
    positions: Quantity
    source_path: Path


def prepare_protein(
    protein_pdb_path: str | Path,
    *,
    add_hydrogens: bool = True,
    ph: float = 7.4,
    forcefield_files: tuple[str, ...] = _DEFAULT_FORCEFIELD_FILES,
    output_dir: str | Path | None = None,
) -> PreparedProtein:
    """Repair a protein PDB with PDBFixer and return a PreparedProtein.

    Runs once per input PDB and the result can be reused across many
    minimization calls without repeating the (slow) preparation step.

    Strategy
    --------
    1. PDBFixer fixes non-standard residues and missing heavy atoms.
    2. The fixed structure is round-tripped through a PDB string so that
       OpenMM normalises atom/chain naming consistently.
    3. **All existing H atoms are stripped** from the round-tripped structure.
       This is the critical step: Maestro-prepared PDBs often carry H atoms
       with non-AMBER names or counts (e.g. H1/H2 instead of H/H2/H3 on an
       N-terminal NH3+). Leaving them in causes ``addHydrogens`` to silently
       skip those residues, and ``createSystem`` then fails with
       "No template found … missing N H atoms".
    4. ``Modeller.addHydrogens`` re-adds **all** H atoms from scratch using
       the AMBER14 forcefield so every residue gets exactly the atoms its
       template requires.

    Parameters
    ----------
    protein_pdb_path:
        Path to the input protein PDB file.
    add_hydrogens:
        If True, strip all existing H and re-add them via
        ``Modeller.addHydrogens`` at *ph*.
    ph:
        pH used when determining protonation states (default 7.4).
    forcefield_files:
        Force-field XML files passed to ``ForceField`` for hydrogen addition.
    output_dir:
        Directory to look for/cache the prepared protein PDB. If provided,
        utilizes SHA256 of the input to avoid redundant PDBFixer runs.
    """
    if not _PDBFIXER_AVAILABLE:
        raise ImportError(
            "pdbfixer is required for protein preparation. "
            "Add it to your environment: conda install -c conda-forge pdbfixer"
        )

    path = Path(protein_pdb_path)
    content = path.read_bytes()
    
    hasher = hashlib.sha256()
    hasher.update(content)
    hasher.update(str(add_hydrogens).encode())
    hasher.update(str(ph).encode())
    hasher.update(str(forcefield_files).encode())
    file_hash = hasher.hexdigest()

    cached_pdb_path = None
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        cached_pdb_path = out_path / f"{file_hash}_prepared_protein.pdb"
        if cached_pdb_path.exists():
            # Load from cache
            parsed = PDBFile(str(cached_pdb_path))
            return PreparedProtein(
                topology=parsed.topology,
                positions=parsed.positions,
                source_path=path,
            )

    # --- Step 1: PDBFixer heavy-atom repair ---
    fixer = PDBFixer(filename=str(path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    # Intentionally skip fixer.addMissingHydrogens() — we handle H below.

    # --- Step 2: Round-trip through a PDB string to normalise naming ---
    buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, buf)
    buf.seek(0)
    normalised = PDBFile(buf)

    # --- Step 3: Strip all existing H atoms ---
    # Maestro (and other preparators) may produce H atoms whose names or
    # counts don't match any AMBER14 template variant (e.g. two H atoms on
    # an N-terminal NH2 when AMBER needs three for NH3+).  With those atoms
    # present, addHydrogens silently skips the residue, leaving it broken.
    modeller = Modeller(normalised.topology, normalised.positions)
    if add_hydrogens:
        h_atoms = [
            atom
            for atom in modeller.topology.atoms()
            if atom.element is not None and atom.element.symbol == "H"
        ]
        modeller.delete(h_atoms)

        # --- Step 4: Re-add all H atoms from scratch ---
        # With no existing H, addHydrogens determines terminal/internal status
        # purely from chain position and adds exactly what each template needs.
        ff = ForceField(*forcefield_files)
        modeller.addHydrogens(ff, pH=ph)

    if cached_pdb_path:
        with open(cached_pdb_path, 'w') as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)

    return PreparedProtein(
        topology=modeller.topology,
        positions=modeller.positions,
        source_path=path,
    )
