from __future__ import annotations

import csv
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDConfig


_log = logging.getLogger(__name__)

# Instantiate globally to compile salt SMARTS patterns only once
_SALT_REMOVER = SaltRemover()

@dataclass(frozen=True)
class LigandBond:
    begin: int
    end: int
    order: int

    def to_dict(self) -> dict[str, int]:
        return {"begin": self.begin, "end": self.end, "order": self.order}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LigandBond":
        return cls(
            begin=int(data["begin"]),
            end=int(data["end"]),
            order=int(data.get("order", 1)),
        )


@dataclass(frozen=True)
class PreparedLigandConformer:
    conformer_id: int
    all_atom_coords: npt.NDArray[np.float64]
    atom_symbols: tuple[str, ...]
    bonds: tuple[LigandBond, ...]
    heavy_atom_indices: tuple[int, ...]
    shape_atom_radii: npt.NDArray[np.float64]
    mol_block: str

    @property
    def shape_atom_coords(self) -> npt.NDArray[np.float64]:
        return self.all_atom_coords[np.asarray(self.heavy_atom_indices, dtype=int)]

    @property
    def centroid(self) -> npt.NDArray[np.float64]:
        return self.shape_atom_coords.mean(axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conformer_id": self.conformer_id,
            "all_atom_coords": self.all_atom_coords.tolist(),
            "atom_symbols": list(self.atom_symbols),
            "bonds": [bond.to_dict() for bond in self.bonds],
            "heavy_atom_indices": list(self.heavy_atom_indices),
            "shape_atom_radii": self.shape_atom_radii.tolist(),
            "color_points": [point.to_dict() for point in self.color_points],
            "mol_block": self.mol_block,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedLigandConformer":
        return cls(
            conformer_id=int(data["conformer_id"]),
            all_atom_coords=np.asarray(data["all_atom_coords"], dtype=np.float64),
            atom_symbols=tuple(str(symbol) for symbol in data["atom_symbols"]),
            bonds=tuple(LigandBond.from_dict(bond) for bond in data["bonds"]),
            heavy_atom_indices=tuple(int(idx) for idx in data["heavy_atom_indices"]),
            shape_atom_radii=np.asarray(data["shape_atom_radii"], dtype=np.float64),
            mol_block=str(data["mol_block"]),
        )


@dataclass(frozen=True)
class PreparedLigand:
    name: str
    canonical_smiles: str
    source: str
    conformers: tuple[PreparedLigandConformer, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_smiles": self.canonical_smiles,
            "source": self.source,
            "conformers": [conformer.to_dict() for conformer in self.conformers],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedLigand":
        return cls(
            name=str(data["name"]),
            canonical_smiles=str(data["canonical_smiles"]),
            source=str(data["source"]),
            conformers=tuple(
                PreparedLigandConformer.from_dict(conformer)
                for conformer in data["conformers"]
            ),
        )


def prepare_ligand_from_smiles(
    smiles: str,
    *,
    name: str | None = None,
    num_conformers: int = 10,
    prune_rms_threshold: float = 1.0,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
    enumerate_stereoisomers: bool = True,
    max_stereoisomers: int = 16,
) -> PreparedLigand:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    prepared_name = name or smiles
    return _prepare_ligand_mol(
        mol,
        source=f"smiles:{smiles}",
        name=prepared_name,
        num_conformers=num_conformers,
        prune_rms_threshold=prune_rms_threshold,
        random_seed=random_seed,
        optimize=optimize,
        max_iterations=max_iterations,
        energy_cutoff=energy_cutoff,
        enumerate_tautomers=enumerate_tautomers,
        max_tautomers=max_tautomers,
        enumerate_stereoisomers=enumerate_stereoisomers,
        max_stereoisomers=max_stereoisomers,
    )


def prepare_ligand_from_file(
    ligand_path: str | os.PathLike[str],
    *,
    name: str | None = None,
    num_conformers: int = 10,
    prune_rms_threshold: float = 1.0,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
    enumerate_stereoisomers: bool = True,
    max_stereoisomers: int = 16,
) -> PreparedLigand:
    path = Path(ligand_path)
    mol = _load_molecule_from_file(path)
    prepared_name = name or path.stem
    return _prepare_ligand_mol(
        mol,
        source=str(path),
        name=prepared_name,
        num_conformers=num_conformers,
        prune_rms_threshold=prune_rms_threshold,
        random_seed=random_seed,
        optimize=optimize,
        max_iterations=max_iterations,
        energy_cutoff=energy_cutoff,
        enumerate_tautomers=enumerate_tautomers,
        max_tautomers=max_tautomers,
        enumerate_stereoisomers=enumerate_stereoisomers,
        max_stereoisomers=max_stereoisomers,
    )


def save_prepared_ligand(
    prepared_ligand: PreparedLigand,
    output_path: str | os.PathLike[str],
) -> None:
    path = Path(output_path)
    path.write_text(json.dumps(prepared_ligand.to_dict(), indent=2))


def load_prepared_ligand(
    input_path: str | os.PathLike[str],
) -> PreparedLigand:
    path = Path(input_path)
    return PreparedLigand.from_dict(json.loads(path.read_text()))


def _prepare_csv_row(
    row: dict[str, str],
    num_conformers: int,
    prune_rms_threshold: float,
    random_seed: int,
    energy_cutoff: float,
    enumerate_tautomers: bool,
    max_tautomers: int,
    enumerate_stereoisomers: bool,
    max_stereoisomers: int,
) -> tuple[bool | None, str, PreparedLigand | None, str | None]:
    name = row["name"].strip()
    smiles = row["smiles"].strip()
    activity_raw = str(row["active"]).strip()
    if activity_raw == "1":
        is_active: bool | None = True
    elif activity_raw == "0":
        is_active = False
    else:
        return None, name, None, f"invalid activity value '{activity_raw}' (expected 0 or 1)"

    try:
        prep = prepare_ligand_from_smiles(
            smiles, name=name,
            num_conformers=num_conformers,
            prune_rms_threshold=prune_rms_threshold,
            random_seed=random_seed,
            energy_cutoff=energy_cutoff,
            enumerate_tautomers=enumerate_tautomers,
            max_tautomers=max_tautomers,
            enumerate_stereoisomers=enumerate_stereoisomers,
            max_stereoisomers=max_stereoisomers,
        )
        return is_active, name, prep, None
    except Exception as e:
        return is_active, name, None, str(e)


def _prepare_screening_csv_row(
    row: dict[str, str],
    num_conformers: int,
    prune_rms_threshold: float,
    random_seed: int,
    energy_cutoff: float,
    enumerate_tautomers: bool,
    max_tautomers: int,
    enumerate_stereoisomers: bool,
    max_stereoisomers: int,
) -> tuple[str, PreparedLigand | None, str | None]:
    name = row["name"].strip()
    smiles = row["smiles"].strip()

    try:
        prep = prepare_ligand_from_smiles(
            smiles,
            name=name,
            num_conformers=num_conformers,
            prune_rms_threshold=prune_rms_threshold,
            random_seed=random_seed,
            energy_cutoff=energy_cutoff,
            enumerate_tautomers=enumerate_tautomers,
            max_tautomers=max_tautomers,
            enumerate_stereoisomers=enumerate_stereoisomers,
            max_stereoisomers=max_stereoisomers,
        )
        return name, prep, None
    except Exception as e:
        return name, None, str(e)


def load_compound_csv(
    csv_path: str | os.PathLike[str],
    *,
    num_conformers: int = 10,
    prune_rms_threshold: float = 1.0,
    random_seed: int = 0xF00D,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
    enumerate_stereoisomers: bool = True,
    max_stereoisomers: int = 16,
    name_column: str = "name",
    smiles_column: str = "smiles",
    activity_column: str = "active",
    on_prepared: Callable[[PreparedLigand, str], None] | None = None,
) -> tuple[list[PreparedLigand], list[PreparedLigand], dict[str, str]]:
    """Load a compound CSV and return (actives, inactives, name_map).

    The returned ``name_map`` maps each state-level name back to the parent
    compound name (identity mapping now that protonation states are removed).

    CSV format (required columns by default):
        name,smiles,active
        compound_A,CCOc1ccc(...)cc1,1
        inactive_1,CCCCCC,0
    Rows with ``active`` values other than ``0`` or ``1`` are skipped with a warning.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    num_conformers / prune_rms_threshold / random_seed / energy_cutoff:
        Forwarded to ligand preparation.
    name_column / smiles_column / activity_column:
        CSV column names to use for parent ligand name, SMILES, and activity
        label respectively.
    on_prepared:
        Optional callback invoked in the parent process for each successfully
        prepared ligand as ``on_prepared(prepared_ligand, parent_name)``.

    Returns
    -------
    tuple[list[PreparedLigand], list[PreparedLigand], dict[str, str]]
        ``(actives, inactives, name_map)``  where ``name_map`` maps
        ``state_name → parent_name``.
    """
    path = Path(csv_path)
    actives: list[PreparedLigand] = []
    inactives: list[PreparedLigand] = []
    name_map: dict[str, str] = {}  # state_name → parent_name

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Compound CSV {path} has no header row.")
        required = {name_column, smiles_column, activity_column}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Compound CSV {path} must have columns: {required}. "
                f"Found: {reader.fieldnames}"
            )
        rows = [
            {
                "name": str(row[name_column]),
                "smiles": str(row[smiles_column]),
                "active": str(row[activity_column]),
            }
            for row in reader
        ]

    import concurrent.futures
    import functools
    import os

    worker = functools.partial(
        _prepare_csv_row,
        num_conformers=num_conformers,
        prune_rms_threshold=prune_rms_threshold,
        random_seed=random_seed,
        energy_cutoff=energy_cutoff,
        enumerate_tautomers=enumerate_tautomers,
        max_tautomers=max_tautomers,
        enumerate_stereoisomers=enumerate_stereoisomers,
        max_stereoisomers=max_stereoisomers,
    )

    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    # Leave 1 CPU free to avoid complete starvation
    max_workers = max(1, max_workers - 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for is_active, name, prep, err in executor.map(worker, rows):
            if err is not None:
                if is_active is None:
                    _log.warning("Skipping ligand '%s': %s", name, err)
                else:
                    _log.warning("Failed to prepare ligand '%s': %s", name, err)
                continue
            assert prep is not None
            assert is_active is not None
            name_map[prep.name] = name
            if on_prepared is not None:
                on_prepared(prep, name)
            if is_active:
                actives.append(prep)
            else:
                inactives.append(prep)

    return actives, inactives, name_map


def load_screening_csv(
    csv_path: str | os.PathLike[str],
    *,
    num_conformers: int = 10,
    prune_rms_threshold: float = 1.0,
    random_seed: int = 0xF00D,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
    enumerate_stereoisomers: bool = True,
    max_stereoisomers: int = 16,
    name_column: str = "name",
    smiles_column: str = "smiles",
    on_prepared: Callable[[PreparedLigand, str], None] | None = None,
) -> tuple[list[PreparedLigand], dict[str, str]]:
    """Load an unlabeled screening CSV and return (ligands, name_map).

    CSV format (required columns by default):
        name,smiles
        compound_A,CCOc1ccc(...)cc1
        compound_B,CCCCCC

    Returns
    -------
    tuple[list[PreparedLigand], dict[str, str]]
        ``(ligands, name_map)`` where ``name_map`` maps
        ``state_name → parent_name``.

    Parameters
    ----------
    name_column / smiles_column:
        CSV column names to use for parent ligand name and SMILES.
    on_prepared:
        Optional callback invoked in the parent process for each successfully
        prepared ligand as ``on_prepared(prepared_ligand, parent_name)``.
    """
    path = Path(csv_path)
    ligands: list[PreparedLigand] = []
    name_map: dict[str, str] = {}

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Screening CSV {path} has no header row.")
        required = {name_column, smiles_column}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Screening CSV {path} must have columns: {required}. "
                f"Found: {reader.fieldnames}"
            )
        rows = [
            {
                "name": str(row[name_column]),
                "smiles": str(row[smiles_column]),
            }
            for row in reader
        ]

    import concurrent.futures
    import functools
    import os

    worker = functools.partial(
        _prepare_screening_csv_row,
        num_conformers=num_conformers,
        prune_rms_threshold=prune_rms_threshold,
        random_seed=random_seed,
        energy_cutoff=energy_cutoff,
        enumerate_tautomers=enumerate_tautomers,
        max_tautomers=max_tautomers,
        enumerate_stereoisomers=enumerate_stereoisomers,
        max_stereoisomers=max_stereoisomers,
    )

    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    max_workers = max(1, max_workers - 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for name, prep, err in executor.map(worker, rows):
            if err is not None:
                _log.warning("Failed to prepare ligand '%s': %s", name, err)
                continue
            assert prep is not None
            name_map[prep.name] = name
            if on_prepared is not None:
                on_prepared(prep, name)
            ligands.append(prep)

    return ligands, name_map


def write_ligand_sdf(
    prepared_ligand: PreparedLigand,
    output_path: str | os.PathLike[str],
) -> Path:
    """Write all conformers of a PreparedLigand to a multi-entry SDF file.

    GNINA reads multi-conformer SDF natively; each conformer is written as a
    separate SDF entry so GNINA can choose the best starting pose.

    The molecule name (first line of each MOL block) is set to the ligand
    name, and SD properties ``_Name`` and ``SMILES`` are included.

    Parameters
    ----------
    prepared_ligand:
        A ``PreparedLigand`` with one or more conformers.
    output_path:
        Path to write the output ``.sdf`` file.

    Returns
    -------
    Path
        The resolved path of the written file.
    """
    out = Path(output_path)
    writer = Chem.SDWriter(str(out))
    for conformer in prepared_ligand.conformers:
        mol = Chem.MolFromMolBlock(conformer.mol_block, removeHs=False)
        if mol is None:
            _log.warning(
                "Failed to reconstruct mol from mol_block for conformer %d of '%s'. Skipping.",
                conformer.conformer_id, prepared_ligand.name,
            )
            continue
        mol.SetProp("_Name", prepared_ligand.name)
        mol.SetProp("SMILES", prepared_ligand.canonical_smiles)
        writer.write(mol)
    writer.close()
    return out


def write_ligands_for_docking(
    compounds: list[PreparedLigand],
    ligands_dir: str | os.PathLike[str],
) -> dict[str, str]:
    """Write all prepared ligands as SDF files for GNINA docking.

    Parameters
    ----------
    compounds:
        List of ``PreparedLigand`` objects (may include multiple protonation
        states per parent compound).
    ligands_dir:
        Directory to write SDF files into.

    Returns
    -------
    dict[str, str]
        Mapping ``{state_name: absolute_sdf_path}``.
    """
    out_dir = Path(ligands_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for lig in compounds:
        sdf_path = write_ligand_sdf(lig, out_dir / f"{lig.name}.sdf")
        paths[lig.name] = str(sdf_path.resolve())
    return paths


def conformer_to_pdb_block(
    conformer: PreparedLigandConformer,
    coords: npt.NDArray[np.float64] | None = None,
    *,
    residue_name: str = "LIG",
    chain_id: str = "L",
    residue_id: int = 1,
) -> str:
    coords = conformer.all_atom_coords if coords is None else np.asarray(coords, dtype=float)
    if coords.shape != conformer.all_atom_coords.shape:
        raise ValueError(
            "coords must have shape "
            f"{conformer.all_atom_coords.shape}, got {coords.shape}"
        )

    lines: list[str] = []
    for idx, (symbol, (x, y, z)) in enumerate(zip(conformer.atom_symbols, coords), start=1):
        atom_name = f"{symbol}{idx % 1000:03d}"[:4]
        lines.append(
            "HETATM"
            f"{idx:5d} "
            f"{atom_name:<4}"
            f" {residue_name:>3} "
            f"{chain_id:1}"
            f"{residue_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}          "
            f"{symbol:>2}"
        )

    for bond in conformer.bonds:
        lines.append(f"CONECT{bond.begin + 1:5d}{bond.end + 1:5d}")

    lines.append("END")
    return "\n".join(lines) + "\n"


def _prepare_ligand_mol(
    mol: Mol,
    *,
    source: str,
    name: str,
    num_conformers: int,
    prune_rms_threshold: float,
    random_seed: int,
    optimize: bool,
    max_iterations: int,
    energy_cutoff: float,
    enumerate_tautomers: bool,
    max_tautomers: int,
    enumerate_stereoisomers: bool,
    max_stereoisomers: int,
) -> PreparedLigand:
    mol = _strip_salts(mol)
    
    tautomer_mols = _enumerate_tautomers_limited(mol, enumerate_tautomers, max_tautomers)
    canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    prepared_conformers: list[PreparedLigandConformer] = []
    state_index = 0

    for tautomer in tautomer_mols:
        stereoisomer_mols = _enumerate_stereoisomers_limited(
            tautomer, enumerate_stereoisomers, max_stereoisomers
        )
        for stereoisomer in stereoisomer_mols:
            mol_with_h = Chem.AddHs(Chem.Mol(stereoisomer))

            conformer_ids = _embed_conformers(
                mol_with_h, num_conformers, random_seed + state_index
            )
            state_index += 1
            if not conformer_ids:
                continue

            if optimize:
                energies = _optimize_conformers(mol_with_h, conformer_ids, max_iterations)
            else:
                energies = _calc_unoptimized_energies(mol_with_h, conformer_ids)

            keep_cids = _filter_and_prune_conformers(
                mol_with_h, energies, energy_cutoff, prune_rms_threshold
            )

            for conf_id in keep_cids:
                extracted = _extract_conformer(mol_with_h, conf_id)
                prepared_conformers.append(
                    PreparedLigandConformer(
                        conformer_id=len(prepared_conformers),
                        all_atom_coords=extracted.all_atom_coords,
                        atom_symbols=extracted.atom_symbols,
                        bonds=extracted.bonds,
                        heavy_atom_indices=extracted.heavy_atom_indices,
                        shape_atom_radii=extracted.shape_atom_radii,
                        mol_block=extracted.mol_block,
                    )
                )

    if not prepared_conformers:
        raise ValueError(f"RDKit failed to embed conformers for ligand '{name}'")

    return PreparedLigand(
        name=name,
        canonical_smiles=canonical_smiles,
        source=source,
        conformers=tuple(prepared_conformers),
    )


def _load_molecule_from_file(path: Path) -> Mol:
    if path.suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path))
        mol = next(supplier)
    elif path.suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    if mol is None:
        raise ValueError(f"Could not load molecule from {path}")
    return mol


def _strip_salts(mol: Mol) -> Mol:
    return _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)


def _enumerate_tautomers_limited(mol: Mol, enumerate_tautomers: bool, max_tautomers: int) -> list[Mol]:
    if not enumerate_tautomers:
        return [Chem.RemoveHs(Chem.Mol(mol))]
    return _enumerate_tautomer_molecules(mol, max_tautomers=max_tautomers)


def _enumerate_stereoisomers_limited(
    mol: Mol, enumerate_stereoisomers: bool, max_stereoisomers: int
) -> list[Mol]:
    if not enumerate_stereoisomers:
        return [Chem.RemoveHs(Chem.Mol(mol))]
    return _enumerate_stereoisomer_molecules(mol, max_stereoisomers=max_stereoisomers)


def _enumerate_tautomer_molecules(mol: Mol, max_tautomers: int) -> list[Mol]:
    enumerator = rdMolStandardize.TautomerEnumerator()
    # Enumerate generates tautomers; we take up to max_tautomers.
    # Note: Enumerate returns a TautomerEnumeratorResult which is iterable.
    tautomers = enumerator.Enumerate(mol)
    return [t for i, t in enumerate(tautomers) if i < max_tautomers]


def _enumerate_stereoisomer_molecules(mol: Mol, max_stereoisomers: int) -> list[Mol]:
    options = StereoEnumerationOptions(
        tryEmbedding=False,
        unique=True,
        onlyUnassigned=False,
        maxIsomers=max_stereoisomers,
    )
    stereoisomers = list(EnumerateStereoisomers(mol, options=options))
    if not stereoisomers:
        return [Chem.RemoveHs(Chem.Mol(mol))]
    return [Chem.RemoveHs(Chem.Mol(iso)) for iso in stereoisomers]


def _embed_conformers(mol_with_h: Mol, num_conformers: int, seed: int) -> list[int]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0 # Prune later
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    return list(AllChem.EmbedMultipleConfs(mol_with_h, numConfs=num_conformers, params=params))


def _optimize_conformers(mol: Mol, conformer_ids: list[int], max_iterations: int) -> list[tuple[int, float]]:
    energies: list[tuple[int, float]] = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in conformer_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIts=max_iterations)
                energies.append((conf_id, ff.CalcEnergy()))
    else:
        for conf_id in conformer_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIts=max_iterations)
                energies.append((conf_id, ff.CalcEnergy()))
    return energies


def _calc_unoptimized_energies(mol: Mol, conformer_ids: list[int]) -> list[tuple[int, float]]:
    energies: list[tuple[int, float]] = []
    if AllChem.MMFFHasAllMoleculeParams(mol):
        props = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in conformer_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            if ff is not None:
                energies.append((conf_id, ff.CalcEnergy()))
    else:
        for conf_id in conformer_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is not None:
                energies.append((conf_id, ff.CalcEnergy()))
    return energies


def _filter_and_prune_conformers(
    mol: Mol,
    energies: list[tuple[int, float]],
    energy_cutoff: float,
    prune_rms_threshold: float,
) -> list[int]:
    if not energies:
        return []
    
    energies.sort(key=lambda x: x[1])
    min_energy = energies[0][1]
    filtered_cids = [cid for cid, energy in energies if energy - min_energy <= energy_cutoff]

    if prune_rms_threshold <= 0:
        return filtered_cids

    keep_cids: list[int] = []
    for cid in filtered_cids:
        if not keep_cids:
            keep_cids.append(cid)
            continue
        too_close = False
        for kept_cid in keep_cids:
            rmsd = AllChem.GetBestRMS(mol, mol, cid, kept_cid)
            if rmsd < prune_rms_threshold:
                too_close = True
                break
        if not too_close:
            keep_cids.append(cid)
    return keep_cids


def _extract_conformer(mol: Mol, conformer_id: int) -> PreparedLigandConformer:
    conformer = mol.GetConformer(conformer_id)
    periodic_table = Chem.GetPeriodicTable()

    atom_symbols: list[str] = []
    all_atom_coords = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float64)
    heavy_atom_indices: list[int] = []
    heavy_atom_radii: list[float] = []

    for atom_index, atom in enumerate(mol.GetAtoms()):
        position = conformer.GetAtomPosition(atom_index)
        all_atom_coords[atom_index] = [position.x, position.y, position.z]
        atom_symbols.append(atom.GetSymbol())

        if atom.GetAtomicNum() > 1:
            heavy_atom_indices.append(atom_index)
            heavy_atom_radii.append(float(periodic_table.GetRvdw(atom.GetAtomicNum())))

    bonds = tuple(
        LigandBond(
            begin=bond.GetBeginAtomIdx(),
            end=bond.GetEndAtomIdx(),
            order=int(bond.GetBondTypeAsDouble()),
        )
        for bond in mol.GetBonds()
    )
    mol_block = Chem.MolToMolBlock(mol, confId=conformer_id)

    return PreparedLigandConformer(
        conformer_id=conformer_id,
        all_atom_coords=all_atom_coords,
        atom_symbols=tuple(atom_symbols),
        bonds=bonds,
        heavy_atom_indices=tuple(heavy_atom_indices),
        shape_atom_radii=np.asarray(heavy_atom_radii, dtype=np.float64),
        mol_block=mol_block,
    )


def approximate_shape_self_overlap(
    conformer: PreparedLigandConformer,
) -> float:
    return float(
        np.sum((4.0 / 3.0) * np.pi * np.power(conformer.shape_atom_radii, 3))
    )


def compute_molecular_formula(prepared_ligand: PreparedLigand) -> str:
    mol = Chem.MolFromSmiles(prepared_ligand.canonical_smiles)
    if mol is None:
        return ""
    return rdMolDescriptors.CalcMolFormula(mol)


# ─────────────────────────────────────────────────────────────────────────────
# LigPrep-based ligand preparation (for Glide docking)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_schrodinger_binary(name: str) -> str:
    """Resolve a Schrödinger binary path via the ``SCHRODINGER`` env variable."""
    schrodinger = os.environ.get("SCHRODINGER", "").strip()
    if schrodinger:
        from pathlib import Path as _Path
        return str(_Path(schrodinger) / name)
    return name


def run_ligprep(
    smiles: str,
    output_maegz: "os.PathLike[str] | str",
    *,
    name: str | None = None,
    ph: float = 7.0,
    ph_tolerance: float = 1.0,
    max_states: int = 1,
    ligprep_binary: str | None = None,
    timeout_seconds: int | None = 300,
    work_dir: "os.PathLike[str] | str | None" = None,
) -> Path:
    """Run Schrödinger LigPrep on a single SMILES to produce a ``.maegz`` file.

    LigPrep handles protonation, tautomer enumeration, and 3-D conformer
    generation at the specified pH, making it suitable as input for Glide.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    output_maegz:
        Path for the output ``.maegz`` file.
    name:
        Optional molecule name embedded in the output file.
    ph:
        Target pH for protonation (default 7.0).
    ph_tolerance:
        pH range ± around ``ph`` (default 1.0 → pH 6.0–8.0).
    max_states:
        Maximum number of protonation / tautomer states to generate
        (default 1 = single best state).
    ligprep_binary:
        Path or name of the LigPrep binary. Defaults to
        ``$SCHRODINGER/ligprep``.
    timeout_seconds:
        Wall-clock timeout in seconds (default 300).
    work_dir:
        Working directory for LigPrep sidecar logs/files. If None, defaults
        to ``output_maegz.parent``.

    Returns
    -------
    Path
        Resolved path of the written ``.maegz`` file.

    Raises
    ------
    RuntimeError
        If LigPrep fails or produces no output.
    """
    import subprocess
    import tempfile
    from pathlib import Path as _Path

    output_maegz = _Path(output_maegz).resolve()
    output_maegz.parent.mkdir(parents=True, exist_ok=True)
    run_cwd = _Path(work_dir).resolve() if work_dir is not None else output_maegz.parent
    run_cwd.mkdir(parents=True, exist_ok=True)

    ligprep_bin = ligprep_binary or _resolve_schrodinger_binary("ligprep")

    # Write SMILES input file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smi", delete=False, prefix="draco_ligprep_"
    ) as tmp_smi:
        mol_name = name or smiles[:30].replace(" ", "_")
        tmp_smi.write(f"{smiles} {mol_name}\n")
        smi_path = _Path(tmp_smi.name)

    try:
        cmd = [
            ligprep_bin,
            "-ismi", str(smi_path),
            "-omae", str(output_maegz),
            "-ph", str(ph),
            "-pht", str(ph_tolerance),
            "-s", str(max_states),
            "-epik",   # use Epik for protonation state prediction
            "-WAIT",
            "-LOCAL",
        ]
        _log.debug("LigPrep cmd: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(run_cwd),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"LigPrep timed out after {timeout_seconds} s for SMILES '{smiles[:40]}…'"
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"LigPrep binary not found at '{ligprep_bin}'. "
                "Ensure $SCHRODINGER is set (module load schrodinger)."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                f"LigPrep failed (exit {proc.returncode}) for SMILES '{smiles[:40]}':\n"
                f"stdout: {proc.stdout[:400]}\nstderr: {proc.stderr[:400]}"
            )

        if not output_maegz.exists() or output_maegz.stat().st_size == 0:
            raise RuntimeError(
                f"LigPrep produced no output at '{output_maegz}' for SMILES '{smiles[:40]}'."
            )

        return output_maegz
    finally:
        try:
            smi_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ligprep_worker(
    args: tuple[str, str, str, float, float, int, str | None, int | None, str | None],
) -> tuple[str, str | None, str | None]:
    """Worker for parallel LigPrep execution.

    Parameters
    ----------
    args:
        ``(name, smiles, output_maegz_str, ph, ph_tolerance, max_states,
           ligprep_binary, timeout_seconds, work_dir_str)``

    Returns
    -------
    tuple[str, str | None, str | None]
        ``(name, maegz_path_str | None, error_msg | None)``
    """
    name, smiles, output_maegz_str, ph, ph_tolerance, max_states, ligprep_bin, timeout_seconds, work_dir_str = args
    try:
        path = run_ligprep(
            smiles,
            output_maegz_str,
            name=name,
            ph=ph,
            ph_tolerance=ph_tolerance,
            max_states=max_states,
            ligprep_binary=ligprep_bin,
            timeout_seconds=timeout_seconds,
            work_dir=work_dir_str,
        )
        return name, str(path), None
    except Exception as exc:
        return name, None, str(exc)


def prepare_ligands_for_glide(
    compounds: list[PreparedLigand],
    ligands_dir: "os.PathLike[str] | str",
    *,
    ph: float = 7.0,
    ph_tolerance: float = 1.0,
    max_states: int = 1,
    ligprep_binary: str | None = None,
    timeout_seconds: int | None = 300,
    max_workers: int | None = None,
) -> dict[str, str]:
    """Run LigPrep on ligands and return ``.maegz`` paths.

    By default this prepares all ligands from the input set into a single
    LigPrep library file ``ligprep_all.maegz`` under ``ligands_dir`` and
    returns a mapping from each ligand name to that shared file path.

    Parameters
    ----------
    compounds:
        List of ``PreparedLigand`` objects. Only ``canonical_smiles`` and
        ``name`` fields are used; the RDKit conformers are not forwarded
        (LigPrep generates its own 3-D structures).
    ligands_dir:
        Directory in which to write the output ``.maegz`` files.
    ph:
        Target pH for LigPrep protonation (default 7.0).
    ph_tolerance:
        pH range ± around ``ph`` (default 1.0).
    max_states:
        Maximum protonation / tautomer states per ligand (default 1).
    ligprep_binary:
        Path or name of the LigPrep binary. Defaults to ``$SCHRODINGER/ligprep``.
    timeout_seconds:
        Per-ligand LigPrep timeout in seconds (default 300).
    max_workers:
        Reserved for backward compatibility (unused in shared-library mode).

    Returns
    -------
    dict[str, str]
        Mapping ``{ligand_name: absolute_maegz_path}``. In shared-library mode,
        each ligand maps to the same ``ligprep_all.maegz`` path.
    """
    import subprocess

    out_dir = Path(ligands_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ligprep_logs_dir = out_dir / "ligprep_logs"
    ligprep_logs_dir.mkdir(parents=True, exist_ok=True)

    ligprep_bin = ligprep_binary or _resolve_schrodinger_binary("ligprep")
    if not compounds:
        return {}

    # Shared LigPrep input/output for all CSV ligands to keep file count low.
    smi_path = ligprep_logs_dir / "ligprep_input_all.smi"
    output_maegz = (out_dir / "ligprep_all.maegz").resolve()
    stdout_path = ligprep_logs_dir / "ligprep_all.stdout.log"
    stderr_path = ligprep_logs_dir / "ligprep_all.stderr.log"

    lines = []
    for compound in compounds:
        lines.append(f"{compound.canonical_smiles} {compound.name}")
    smi_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ligprep_bin,
        "-ismi", str(smi_path.resolve()),
        "-omae", str(output_maegz),
        "-ph", str(ph),
        "-pht", str(ph_tolerance),
        "-s", str(max_states),
        "-epik",
        "-WAIT",
        "-LOCAL",
    ]
    _log.debug("LigPrep batch cmd: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            # LigPrep requires output paths to be under the process CWD.
            # Run from out_dir so -omae <out_dir>/ligprep_all.maegz is valid.
            cwd=str(out_dir),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"LigPrep batch timed out after {timeout_seconds} s for {len(compounds)} ligands."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"LigPrep binary not found at '{ligprep_bin}'. "
            "Ensure $SCHRODINGER is set (module load schrodinger)."
        )

    try:
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    except OSError as exc:
        _log.warning("Failed to write LigPrep batch logs: %s", exc)

    if proc.returncode != 0:
        raise RuntimeError(
            f"LigPrep batch failed (exit {proc.returncode}) for {len(compounds)} ligands.\n"
            f"stdout: {(proc.stdout or '')[:400]}\nstderr: {(proc.stderr or '')[:400]}"
        )

    if not output_maegz.exists() or output_maegz.stat().st_size == 0:
        raise RuntimeError(
            f"LigPrep batch produced no output at '{output_maegz}'."
        )

    shared_path = str(output_maegz)
    return {compound.name: shared_path for compound in compounds}