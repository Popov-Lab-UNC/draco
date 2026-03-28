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
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDConfig

_log = logging.getLogger(__name__)

# Instantiate globally to compile salt SMARTS patterns only once
_SALT_REMOVER = SaltRemover()

_FEATURE_LABEL_MAP = {
    "Donor": "donor",
    "Acceptor": "acceptor",
    "PosIonizable": "cation",
    "NegIonizable": "anion",
    "Aromatic": "ring",
    # "Hydrophobe": "hydrophobe",  # Removed to prevent the "hydrophobic swarm"
    "LumpedHydrophobe": "hydrophobe",
}


@dataclass(frozen=True)
class LigandColorPoint:
    label: str
    coords: npt.NDArray[np.float64]
    atom_indices: tuple[int, ...]
    radius: float = 1.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "coords": self.coords.tolist(),
            "atom_indices": list(self.atom_indices),
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LigandColorPoint":
        return cls(
            label=str(data["label"]),
            coords=np.asarray(data["coords"], dtype=np.float64),
            atom_indices=tuple(int(idx) for idx in data["atom_indices"]),
            radius=float(data.get("radius", 1.7)),
        )


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
    color_points: tuple[LigandColorPoint, ...]
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
            color_points=tuple(
                LigandColorPoint.from_dict(point) for point in data["color_points"]
            ),
            mol_block=str(data["mol_block"]),
        )


@dataclass(frozen=True)
class SphereOverlapResult:
    """Per-sphere compatibility result from ligand–pocket feature overlap scoring."""

    sphere_id: int
    sphere_labels: tuple[str, ...]
    matched_ligand_label: str
    ligand_point_index: int
    distance: float
    compatibility: int


@dataclass(frozen=True)
class PocketAlignPoint:
    """Internal: pocket feature anchor for rigid-body alignment to a ligand."""

    feature_id: str
    label: str
    coords: npt.NDArray[np.float64]
    sphere_id: int


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


def protonate_smiles(
    smiles: str,
    *,
    ph_min: float = 6.4,
    ph_max: float = 8.4,
    max_variants: int = 8,
    precision: float = 1.0,
) -> list[str]:
    """Generate protonation states of a SMILES at a given pH range using Dimorphite-DL.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    ph_min / ph_max:
        pH range (default 6.4–8.4 for physiological).
    max_variants:
        Maximum number of protonation variants to return.
    precision:
        pKa precision for Dimorphite-DL (default 1.0).

    Returns
    -------
    list[str]
        List of protonated SMILES strings. If Dimorphite-DL fails or returns
        nothing, falls back to the original SMILES.
    """
    try:
        from dimorphite_dl import protonate_smiles as _dimorphite_protonate
        result = _dimorphite_protonate(
            smiles,
            ph_min=ph_min,
            ph_max=ph_max,
            max_variants=max_variants,
            precision=precision,
            label_states=False,
        )
        if result:
            return list(result)
    except Exception as exc:
        _log.warning("Dimorphite-DL failed for '%s': %s. Using original SMILES.", smiles, exc)
    return [smiles]


def prepare_ligand_from_smiles(
    smiles: str,
    *,
    name: str | None = None,
    num_conformers: int = 20,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
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
    )


def prepare_protonation_states(
    smiles: str,
    *,
    name: str,
    ph_min: float = 6.4,
    ph_max: float = 8.4,
    max_variants: int = 8,
    num_conformers: int = 20,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
) -> list[PreparedLigand]:
    """Protonate a SMILES with Dimorphite-DL, then prepare each state.

    Each protonation state gets its own ``PreparedLigand`` with a name like
    ``'{name}_s0'``, ``'{name}_s1'``, etc.  If only one state is produced
    the name is kept as-is (no suffix).

    Parameters
    ----------
    smiles:
        Input SMILES.
    name:
        Base compound name.
    ph_min / ph_max / max_variants:
        Forwarded to ``protonate_smiles``.
    num_conformers / prune_rms_threshold / random_seed / optimize / max_iterations:
        Forwarded to conformer generation.

    Returns
    -------
    list[PreparedLigand]
        One ``PreparedLigand`` per protonation state.
    """
    protonated = protonate_smiles(
        smiles, ph_min=ph_min, ph_max=ph_max,
        max_variants=max_variants,
    )
    results: list[PreparedLigand] = []
    use_suffix = len(protonated) > 1
    for idx, prot_smi in enumerate(protonated):
        state_name = f"{name}_s{idx}" if use_suffix else name
        try:
            prep = prepare_ligand_from_smiles(
                prot_smi,
                name=state_name,
                num_conformers=num_conformers,
                prune_rms_threshold=prune_rms_threshold,
                random_seed=random_seed,
                optimize=optimize,
                max_iterations=max_iterations,
            )
            results.append(prep)
        except Exception as exc:
            _log.warning(
                "Failed to prepare protonation state %d ('%s') of '%s': %s",
                idx, prot_smi, name, exc,
            )
    if not results:
        # Fallback: prepare original SMILES without protonation
        _log.warning("All protonation states failed for '%s'. Using original SMILES.", name)
        results.append(prepare_ligand_from_smiles(
            smiles, name=name,
            num_conformers=num_conformers,
            prune_rms_threshold=prune_rms_threshold,
            random_seed=random_seed,
            optimize=optimize, max_iterations=max_iterations,
        ))
    return results


def prepare_ligand_from_file(
    ligand_path: str | os.PathLike[str],
    *,
    name: str | None = None,
    num_conformers: int = 20,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    optimize: bool = True,
    max_iterations: int = 200,
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
    num_conformers: int = 20,
    prune_rms_threshold: float = 0.5,
    random_seed: int = 0xF00D,
    protonate: bool = True,
    ph_min: float = 6.4,
    ph_max: float = 8.4,
) -> tuple[list[PreparedLigand], list[PreparedLigand], dict[str, str]]:
    """Load a compound CSV and return (actives, inactives, name_map).

    When ``protonate=True`` (default), each compound is protonated with
    Dimorphite-DL, producing potentially multiple ``PreparedLigand`` objects
    per row.  The returned ``name_map`` maps each state-level name (e.g.
    ``'compound_A_s0'``) back to the parent compound name (``'compound_A'``).

    CSV format (required columns):
        name,smiles,active
        compound_A,CCOc1ccc(...)cc1,1
        inactive_1,CCCCCC,0

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    num_conformers / prune_rms_threshold / random_seed:
        Forwarded to ligand preparation.
    protonate:
        If True, use Dimorphite-DL to enumerate protonation states.
    ph_min / ph_max:
        pH range for protonation (only used if ``protonate=True``).

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

            if protonate:
                states = prepare_protonation_states(
                    smiles, name=name,
                    ph_min=ph_min, ph_max=ph_max,
                    num_conformers=num_conformers,
                    prune_rms_threshold=prune_rms_threshold,
                    random_seed=random_seed,
                )
            else:
                states = [prepare_ligand_from_smiles(
                    smiles, name=name,
                    num_conformers=num_conformers,
                    prune_rms_threshold=prune_rms_threshold,
                    random_seed=random_seed,
                )]

            for prep in states:
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
) -> PreparedLigand:
    # --- Strip disconnected counter-ions (Na+, Cl-, etc.) ---
    mol = _SALT_REMOVER.StripMol(mol, dontRemoveEverything=True)
    
    mol = Chem.AddHs(Chem.Mol(mol))

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = prune_rms_threshold
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
    if not conformer_ids:
        raise ValueError(f"RDKit failed to embed conformers for ligand '{name}'")

    if optimize:
        _optimize_conformers(mol, conformer_ids, max_iterations=max_iterations)

    canonical_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    conformers = tuple(_extract_conformer(mol, conf_id) for conf_id in conformer_ids)

    return PreparedLigand(
        name=name,
        canonical_smiles=canonical_smiles,
        source=source,
        conformers=conformers,
    )


def _load_molecule_from_file(path: Path) -> Mol:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        mol = next((entry for entry in supplier if entry is not None), None)
    elif suffix in {".mol", ".mol2"}:
        mol = Chem.MolFromMolFile(str(path), removeHs=False)
    elif suffix == ".pdb":
        mol = Chem.MolFromPDBFile(str(path), removeHs=False)
    else:
        raise ValueError(
            f"Unsupported ligand file type '{suffix}'. Use SMILES, SDF, MOL, MOL2, or PDB."
        )

    if mol is None:
        raise ValueError(f"Could not load ligand from file: {path}")
    return mol


def _optimize_conformers(
    mol: Mol,
    conformer_ids: list[int],
    *,
    max_iterations: int,
) -> None:
    if AllChem.MMFFHasAllMoleculeParams(mol):
        for conf_id in conformer_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iterations)
        return

    for conf_id in conformer_ids:
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iterations)


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

    color_points = tuple(_extract_color_points(mol, conformer_id, all_atom_coords))
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
        color_points=color_points,
        mol_block=mol_block,
    )


def _extract_color_points(
    mol: Mol,
    conformer_id: int,
    all_atom_coords: npt.NDArray[np.float64],
) -> list[LigandColorPoint]:
    feature_factory = _build_feature_factory()
    features = feature_factory.GetFeaturesForMol(mol, confId=conformer_id)

    points: list[LigandColorPoint] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for feature in features:
        label = _FEATURE_LABEL_MAP.get(feature.GetFamily())
        if label is None:
            continue

        atom_indices = tuple(sorted(int(idx) for idx in feature.GetAtomIds()))
        dedupe_key = (label, atom_indices)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        coords = np.mean(all_atom_coords[np.asarray(atom_indices, dtype=int)], axis=0)
        points.append(LigandColorPoint(label=label, coords=coords, atom_indices=atom_indices))

    return points


def _build_feature_factory() -> ChemicalFeatures.MolChemicalFeatureFactory:
    feature_path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(feature_path)


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