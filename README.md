# draco

**D**ynamic **R**eceptor **A**ctivity-Guided **C**onformational **O**ptimization

A computational pipeline for SAR-guided conformational sampling and cryptic binding site discovery.

## Workflow

![DRACO Workflow](assets/draco_workflow.png)

## Project Plan

See [`project_plan.md`](project_plan.md) for the full project plan and methodology.

## Pocket Coloring Prototype

The repository now includes two early-stage modules for the alpha-sphere workflow:

- `ligand_preparation.py`: prepares ligands from SMILES or files into 3D conformers plus ligand color/pharmacophore points.
- `pocket_coloring.py`: colors `pocketeer` alpha-spheres from local protein context and computes ROCS-like `ShapeTanimoto`, `ColorTanimoto`, and `TanimotoCombo` scores for ligand overlays.

The notebook [`pocketeer_module.ipynb`](pocketeer_module.ipynb) includes a demo section that runs the new workflow after `pocketeer` pocket detection.

## GNINA scores and CNN affinity

GNINA writes several numeric fields into output SDFs. The important distinction is between **Vina-style energy** and **CNN predicted affinity**:

| SDF property | Meaning | “Better” direction |
|--------------|---------|---------------------|
| `minimizedAffinity` | Vina minimized score (kcal/mol) | More **negative** |
| `CNNscore` | Pose quality (probability-like, ~0–1) | **Higher** |
| `CNNaffinity` | CNN **predicted affinity in pK units** | **Higher** (tighter predicted binding) |

**CNN affinity discrepancy (common confusion).** The name “affinity” and sitting next to a kcal/mol field invites treating `CNNaffinity` like `minimizedAffinity`. It is **not** the same unit or sign convention. Per GNINA maintainers, `CNNaffinity` is a **pK-scale** prediction (e.g. larger values correspond to stronger predicted binding; see [gnina/gnina#259](https://github.com/gnina/gnina/issues/259)). In this repo the parsed value is stored as `GninaDockResult.cnn_affinity`; **ranking, top‑k selection, and SAR when using `cnn_affinity` all treat higher values as better.** That differs from `vina_score` / `minimizedAffinity`, where more negative is better.

If you compare results across literature or older notes, confirm whether authors meant pK (CNN) vs kcal/mol (Vina) before interpreting magnitudes.
