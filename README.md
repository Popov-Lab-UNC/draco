# draco

**D**ynamic **R**eceptor **A**ctivity-Guided **C**onformational **O**ptimization

A computational pipeline for SAR-guided conformational sampling and cryptic binding site discovery.

## Workflow

![DRACO Workflow](assets/draco_workflow.png)

## Docking engines

- **`adgpu_gnina` (default)** — AutoDock-GPU pose search in the Pocketeer box, then GNINA
  `--score_only` rescoring (CNN / Vina fields in the output SDF). Requires `adgpu`,
  `autogrid4`, and Meeko scripts (`mk_prepare_receptor.py`, `mk_prepare_ligand.py`)
  on `PATH` (e.g. `module load autodock-gpu autogrid`), plus GNINA (or Apptainer image
  via `--gnina-binary`). The Python env needs **meeko** (declared in `pyproject.toml`
  pixi dependencies) for parsing DLG outputs.
- **`gnina`** — Full GNINA docking in the search box (slower when exhaustiveness is high).
- **`glide`** — Schrödinger Glide + LigPrep (unchanged).

On a single GPU node, Draco defaults to **one concurrent docking worker** for
`adgpu_gnina` to avoid GPU oversubscription; override with `--max-docking-workers` if needed.
