from __future__ import annotations

import csv
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
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
    num_conformers: int = 100,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
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
    )


def prepare_ligand_from_file(
    ligand_path: str | os.PathLike[str],
    *,
    name: str | None = None,
    num_conformers: int = 100,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
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


def load_compound_csv(
    csv_path: str | os.PathLike[str],
    *,
    num_conformers: int = 100,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    energy_cutoff: float = 5.0,
    enumerate_tautomers: bool = True,
    max_tautomers: int = 4,
) -> tuple[list[PreparedLigand], list[PreparedLigand], dict[str, str]]:
    """Load a compound CSV and return (actives, inactives, name_map).

    The returned ``name_map`` maps each state-level name back to the parent
    compound name (identity mapping now that protonation states are removed).

    CSV format (required columns):
        name,smiles,active
        compound_A,CCOc1ccc(...)cc1,1
        inactive_1,CCCCCC,0

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    num_conformers / prune_rms_threshold / random_seed / energy_cutoff:
        Forwarded to ligand preparation.

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

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "smiles", "active"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Compound CSV {path} must have columns: {required}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            name = row["name"].strip()
            smiles = row["smiles"].strip()
            is_active = str(row["active"]).strip() in ("1", "true", "True", "yes")

            prep = prepare_ligand_from_smiles(
                smiles, name=name,
                num_conformers=num_conformers,
                prune_rms_threshold=prune_rms_threshold,
                random_seed=random_seed,
                energy_cutoff=energy_cutoff,
                enumerate_tautomers=enumerate_tautomers,
                max_tautomers=max_tautomers,
            )

            name_map[prep.name] = name
            if is_active:
                actives.append(prep)
            else:
                inactives.append(prep)

    return actives, inactives, name_map


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
) -> PreparedLigand:
    mol = _strip_salts(mol)
    
    tautomer_mols = _enumerate_tautomers_limited(mol, enumerate_tautomers, max_tautomers)
    canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    prepared_conformers: list[PreparedLigandConformer] = []

    for tautomer_index, tautomer in enumerate(tautomer_mols):
        mol_with_h = Chem.AddHs(Chem.Mol(tautomer))
        
        conformer_ids = _embed_conformers(mol_with_h, num_conformers, random_seed + tautomer_index)
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


def _strip_salts(mol: Mol) -> Mol:
    return _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)


def _enumerate_tautomers_limited(mol: Mol, enumerate_tautomers: bool, max_tautomers: int) -> list[Mol]:
    if not enumerate_tautomers:
        return [Chem.RemoveHs(Chem.Mol(mol))]
    return _enumerate_tautomer_molecules(mol, max_tautomers=max_tautomers)


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
                ff.Minimize(maxIters=max_iterations)
                energies.append((conf_id, ff.CalcEnergy()))
    else:
        for conf_id in conformer_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is not None:
                ff.Minimize(maxIters=max_iterations)
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