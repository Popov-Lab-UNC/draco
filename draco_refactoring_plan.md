# Draco Refactoring & Organization Plan

This document outlines the proposed organization and clean up of the Draco pipeline to transition it into a robust, command-line accessible Python package that follows best software engineering practices.

## 1. Package Structure & Organization

Currently, Python scripts are located in the root directory. To make Draco a proper package (installable via `pip` / `pixi`), we need to move the source code into a dedicated package directory and separate deprecated scripts.

**Proposed Directory Structure:**
```text
draco/
├── README.md
├── project_plan.md
├── pyproject.toml
├── pixi.lock
├── src/
│   └── draco/                 # Main package directory
│       ├── __init__.py
│       ├── cli.py             # Entry point for the `draco` command
│       ├── constants.py       # Global constants codified from scattered scripts
│       ├── dynamics.py        # Core MD engine
│       ├── gnina_docking.py   # GNINA subprocess wrapper
│       ├── ligand_preparation.py
│       ├── protein_preparation.py
│       ├── final_refinement.py
│       ├── sar_scoring.py
│       └── utils.py           # Miscellaneous shared helpers
├── deprecated/                # Moved from root
│   ├── local_minimization.py
│   ├── overlay.py
│   ├── pocket_coloring.py
│   ├── visualization.py
│   ├── test_overlay_minimize.py
│   ├── pocketeer_module.ipynb
│   └── test_overlay_minimize.sbatch
└── tests/                     # Future tests directory
```

## 2. CLI Implementation & Hardware Auto-Detection

We want users to run the pipeline using `draco --input_pdb ...` instead of `pixi run python test_dynamics.py ...`.

**CLI Entry Point (`src/draco/cli.py`):**
*   Create a `main()` function utilizing `argparse`.
*   Add entry points in `pyproject.toml`:
    ```toml
    [project.scripts]
    draco = "draco.cli:main"
    ```

**Hardware Auto-Detection & Workload Splitting:**
Currently, `test_dynamics.py` handles parallelization manually but lacks GPU splitting and hardware auto-detection.
*   **CPU Detection:** Use `os.cpu_count()` or `len(os.sched_getaffinity(0))` on Linux to detect available CPUs. Parse `SLURM_CPUS_PER_TASK` if running on a cluster. Allow override via `--n_cpus`.
*   **GPU Detection:** Use PyTorch's `torch.cuda.device_count()` or parse `CUDA_VISIBLE_DEVICES` or run `nvidia-smi -L` via subprocess to count available GPUs. Allow override via `--n_gpus`.
*   **Workload Split Strategy:**
    *   Allocate **1 GPU** (and a dedicated CPU thread) to the **MD simulation** (`dynamics.py`).
    *   Allocate the **remaining GPUs** to a worker pool for **GNINA docking** (which is heavily accelerated by GPUs).
    *   Allocate **remaining CPUs** for **Pocketeer** (pocket detection) and **SAR scoring**, which are CPU-bound.
    *   If only 1 GPU is available, time-share it or restrict GNINA CPU threads so the MD simulation isn't starved.

## 3. Deprecated Modules & Code

According to the `project_plan.md`, the old alpha-sphere pharmacophore overlay approach has been replaced by GNINA docking.

**Files to move to `deprecated/`:**
*   `overlay.py`: Replaced by GNINA.
*   `pocket_coloring.py`: Deprecated for pose generation (though retained for interpretability later, it should be moved out of the core pipeline for now).
*   `local_minimization.py`: Replaced by `final_refinement.py`.
*   `visualization.py`: Depends on `pocket_coloring.py` and `overlay.py`.
*   `test_overlay_minimize.py` and `test_overlay_minimize.sbatch`.
*   `pocketeer_module.ipynb`: Uses deprecated overlay methods.

## 4. Codifying Constants (`src/draco/constants.py`)

Several scripts contain hardcoded constants that should be centralized. This makes the codebase modular, easier to maintain, and prevents duplication.

**Constants to move:**
*   From `dynamics.py` and `local_minimization.py` / `final_refinement.py`:
    *   `KCAL_PER_MOL_A2_TO_KJ_PER_MOL_NM2 = 418.4`
    *   `_DEFAULT_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")`
    *   `_DEFAULT_WATER_MODEL = "tip3pfb"`
    *   `_SOLVENT_ION_RESNAMES` (HOH, WAT, NA, CL, etc.)
*   From `ligand_preparation.py`:
    *   `_FEATURE_LABEL_MAP`
*   From `pocket_coloring.py` (if keeping for interpretability):
    *   `AROMATIC_RESIDUES`, `HYDROPHOBIC_RESIDUES`, `POSITIVE_RESIDUES`, `NEGATIVE_RESIDUES`, `POLAR_RESIDUES`
    *   `FEATURE_COLORS` and `FEATURE_COMPAT`

## 5. File-by-File Review: Pitfalls & Improvements

### `test_dynamics.py` (To become `cli.py`)
*   **Improvement:** The `_dock_frame_worker` is massive. It should be refactored into smaller, testable functions (e.g., `run_pocketeer()`, `run_docking()`, `compute_sar()`).
*   **Pitfall:** `os.environ` thread limits are hardcoded at the top. This is good for preventing oversubscription, but should be dynamically adjustable based on the `--n_cpus` flag.
*   **Improvement:** Re-implement the `ProcessPoolExecutor` to utilize GPUs properly for GNINA, passing specific `CUDA_VISIBLE_DEVICES` to different workers.

### `dynamics.py`
*   **Improvement:** The MD simulation saves frames dynamically inside the `run_dynamics` while loop. This is stateful and tightly coupled to OpenMM. We could decouple the RMSD check from the OpenMM reporter to allow other simulation engines in the future.
*   **Pitfall:** Hardcoded paths (`outdir = Path(output_dir) if output_dir else Path("dynamics_output")`). Should default to a temporary directory or rely strictly on CLI arguments.

### `gnina_docking.py`
*   **Pitfall:** `dock_ligand` relies on `subprocess.run` and parses GNINA's SDF output by string splitting (`$$$$`). This is fragile. We should use `rdkit.Chem.ForwardSDMolSupplier` to parse the SDF output robustly.
*   **Improvement:** Introduce a mechanism to allocate specific GPUs to GNINA subprocess calls via the `CUDA_VISIBLE_DEVICES` environment variable inside the worker.

### `ligand_preparation.py`
*   **Improvement:** `prepare_ligand_from_smiles` uses RDKit's `EmbedMultipleConfs`. If it fails, it raises a `ValueError`. It should fallback to simpler embedding methods or return a structured error result rather than crashing the pipeline worker.
*   **Pitfall:** Dimorphite-DL protonation is fragile. The `except Exception` block masks underlying errors. Logging should be more explicit.

### `final_refinement.py`
*   **Improvement:** `_sdf_block_to_openmm` writes a PDB block string manually (`_mol_to_pdb_block`). Relying on custom PDB writers is risky (especially with atom naming). Using OpenMM's `PDBFile` or RDKit's built-in `MolToPDBBlock` with proper flavor is safer.
*   **Pitfall:** The OpenFF/GAFF parameterization relies on global caches (`_TEMPLATE_GENERATOR_CACHE`). This is fine for a single process, but might cause memory leaks or multiprocessing issues if workers try to share or serialize it.

### `sar_scoring.py`
*   **Improvement:** The `_roc_auc` function re-implements a simple trapezoidal rule to avoid `sklearn` dependency. While lightweight, handling edge cases (ties in scores) can be tricky. Ensure tie-breaking is handled identically to standard libraries.

## 6. Summary of Action Items

1.  Create the `src/draco/` package structure.
2.  Move `overlay.py`, `pocket_coloring.py`, `local_minimization.py`, `visualization.py`, and related tests to `deprecated/`.
3.  Create `src/draco/constants.py` and consolidate global variables.
4.  Refactor `test_dynamics.py` into `src/draco/cli.py` using `argparse`. Add `--n_gpus` and `--n_cpus` arguments.
5.  Implement hardware detection and workload distribution in `cli.py` (MD on GPU 0, GNINA distributed across GPUs 1..N).
6.  Fix brittle parsing in `gnina_docking.py` using RDKit.
7.  Update imports across all remaining core files.
8.  Update `pyproject.toml` to register the CLI command and point to the `src` directory layout.
