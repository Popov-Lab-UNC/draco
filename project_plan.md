# Project Plan 10mar2026

### Summary: SAR-Guided Conformational Sampling for Binding Site Discovery

**The Problem**

Drug discovery campaigns frequently identify hit compounds and establish structure-activity relationships (SAR), yet fail to produce credible binding poses through conventional docking. This failure is not always a ligand problem — it is often a protein problem. The receptor is flexible, the relevant binding conformation is not captured in available crystal structures, and cryptic or allosteric pockets may only open in the presence of a ligand or under particular dynamic conditions. Standard flexible docking methods insufficiently address this because they treat protein flexibility as a local refinement problem rather than a global conformational sampling problem.

The deeper issue is that the field has tools for each sub-problem in isolation, but no unified framework that brings them together in a physically coherent way.

**The Core Vision** 

A unified, open-source framework that accepts:

- A protein structure (PDB, apo or holo)
- A ligand series with associated activity information (actives/inactives, ranked potency, or IC50s)

And produces:

- A protein conformation (and associated pocket geometry) consistent with the SAR data
- Validated binding poses for active compounds
- An explanation of why inactive compounds fail to bind
- A receptor conformation suitable for downstream rigid docking campaigns

The binding site may be known or unknown. If unknown, the framework should operate in a blind mode capable of detecting cryptic or allosteric sites through conformational sampling.

**Key Requirements**

1. **Conformational sampling must be exhaustive and physically meaningful:** The protein cannot be treated as rigid or semi-rigid. The framework needs to explore the genuine conformational ensemble of the receptor, including states that are not represented in any experimental structure.
2. **The activity data must actively guide the search:** This is the central novelty. Actives and inactives are not just validation data — they are a discriminative signal that constrains which protein conformations and binding modes are physically plausible. A conformation that docks actives well but also docks inactives equally well is not a valid solution.
3. **Inactives must be treated as first-class constraints:** No existing tool uses inactives as negative constraints on rotamer assignment or pocket geometry. The framework should use inactive compounds to restrict the conformational and rotamer space — explaining failures is as important as explaining successes.
4. **The scoring function is open:** Empirical docking scores are fast but lack statistical mechanics grounding. Free energy-based methods (such as GCMC occupancy, SILCS FragMaps, or funnel metadynamics) are more rigorous but more expensive. The framework should be agnostic to this choice; the architecture should support both, with the understanding that rigor and computational cost exist on a spectrum and the appropriate choice depends on the campaign stage.
5. **The output must be practically useful:** The end product is not just a scientific result — it is a receptor conformation that a medicinal chemist can use in a rigid docking campaign, or pass directly into FEP for SAR expansion. Interpretability matters: which residues are SAR-determining, which contacts distinguish actives from inactives, and what the pocket geometry looks like.

**What Makes This Novel** 

Every component of this framework has precedent in existing tools. What does not exist is the closed feedback loop between conformational sampling, binding site detection, and SAR-discriminative scoring, particularly the use of inactive compounds as a thermodynamic and structural constraint.

Critically, this approach is **physics-based rather than data-driven**. Recent deep learning methods such as FlexSBDD (NeurIPS 2024) model flexible protein-ligand complexes via generative models, but they are fundamentally dependent on the distribution of known crystal structures for training. They cannot discover conformations that fall outside the training data — precisely the regime where cryptic and allosteric pockets live. Our framework explores conformational space through MD simulation and scores states against experimental SAR data, making it capable of discovering novel states that have never been crystallized.

**Relevant Benchmarking Work**

Bowman et al. (2026) recently benchmarked AI-based methods (AlphaFlow, BioEmu, PocketMiner, CryptoBank) against physics-based MD simulations for cryptic pocket discovery ([bioRxiv 10.64898/2026.01.21.700870](https://www.biorxiv.org/content/10.64898/2026.01.21.700870v1)). Their key finding is that AI methods are fast but qualitative, while simulations are quantitatively predictive but expensive. This directly motivates our approach: use fast geometric triage to make physics-based sampling tractable, and use SAR data to focus sampling on druggable states. The benchmark systems in Bowman et al. (TEM-1 β-lactamase, IL-2, p38α MAPK, etc.) are excellent validation targets for this framework.

---

### The Alpha-Sphere + GNINA Docking Pipeline (Primary Method)

This is the core method for Paper 1. It combines physics-based conformational sampling with fast GPU-accelerated docking to identify SAR-consistent protein conformations and validated binding poses.

#### Overview

The pipeline has **two parallel preparation tracks** that converge at a **docking step**:

- **Track A (Protein side):** Run MD → extract frames → cluster → detect pockets via alpha-sphere tessellation → define docking boxes from pocket geometry
- **Track B (Ligand side):** Take active and inactive compounds → prepare for docking (RDKit conformer generation → SDF output)
- **Convergence:** For each (conformation, pocket) pair, dock all actives and inactives using GNINA (GPU-accelerated, CNN-scored docking). Compute SAR-discrimination scores from docking results. Rank conformations.

#### Step 1: Conformational Sampling via Parallel MD

Run multiple independent MD simulations (e.g. 10 parallel replicas, 100 ns each) from the input structure using OpenMM (explicit solvent, GPU-accelerated). During the simulation:

- Extract frames at regular intervals (e.g. every 100 ps → 1,000 frames per replica → 10,000 total frames)
- **On-the-fly clustering:** Maintain a running set of conformational clusters using backbone RMSD. As each new frame is produced, compare it to existing cluster centroids. If it falls within a threshold (e.g. 2.0 Å RMSD), assign it to the nearest cluster. If not, start a new cluster. This avoids storing and post-processing thousands of frames — the clustering happens concurrently with simulation.
- **Cross-replica comparison:** Periodically share cluster centroids across parallel replicas to avoid redundant exploration. If all replicas are converging on the same clusters, the sampling can be considered exhausted for that timescale.

This concurrent approach allows constant comparison against existing clusters and keeps the number of unique conformations manageable (typically 50–200 distinct states).

#### Step 2: Pocket Detection via Alpha-Sphere Tessellation

For each cluster centroid, run pocket detection using [Pocketeer](https://pocketeer.readthedocs.io/en/latest/) (a modern Python reimplementation of the Fpocket algorithm, installable via pip, built on Biotite atom arrays with Numba JIT acceleration):

1. **Delaunay Tessellation:** Compute the Delaunay triangulation of all protein heavy atoms
2. **Alpha-Sphere Extraction:** For each tetrahedron in the triangulation, compute the circumsphere. Retain spheres with radii between ~3.4 Å and ~6.2 Å (the range that captures drug-sized cavities)
3. **Polarity Labeling:** Classify each alpha-sphere as buried (interior) or surface-exposed based on neighboring atom contacts
4. **Clustering:** Group buried alpha-spheres into discrete pockets using graph connectivity
5. **Scoring:** Rank pockets by volume, burial fraction, and geometric features

This produces a set of alpha-sphere clouds for each conformation, where each cloud represents a potential binding pocket. A single Pocketeer call takes ~10–50 ms per frame, making it feasible to screen thousands of conformations.

#### Step 3: Docking Box Definition from Alpha-Sphere Geometry

For each detected pocket, compute the docking box parameters directly from the alpha-sphere cloud:

- **Center:** Centroid of the alpha-sphere centers
- **Size:** Range of alpha-sphere coordinates along each axis + padding (~4 Å per side)

This replaces the previous pharmacophore coloring step. The alpha-sphere cloud implicitly encodes the pocket shape and volume; the docking engine handles chemistry internally.

> **Note:** Physicochemical coloring of alpha-spheres (H-bond donor/acceptor, hydrophobic, aromatic classification) is retained as an **interpretability layer** for downstream SAR explanation, but is no longer load-bearing for pose generation.

#### Step 4: Ligand Preparation for Docking

For the active and inactive compound sets:

1. **Generate 3D conformers** using RDKit's ETKDG (e.g. 20 conformers per molecule)
2. **Write SDF files** for each compound (GNINA reads SDF/MOL2 natively)
3. **Cache preparations** — this needs to be done once per compound series

#### Step 5: SAR-Discriminative Docking via GNINA

For each (conformation, pocket) pair, dock **all** actives and inactives using [GNINA](https://github.com/gnina/gnina) (GPU-accelerated molecular docking with CNN rescoring):

1. **Docking:** For each compound, dock into the pocket defined by the alpha-sphere docking box. GNINA performs Monte Carlo sampling + local optimization, naturally respecting molecular topology (no ring-threading or steric violation artifacts).

2. **Scoring:** GNINA provides both a Vina-like empirical score and a CNN-based binding affinity prediction. Use the **CNN affinity score** as the primary ranking metric.

3. **SAR-Discriminative Ranking:** For each conformation, rank all molecules (actives + inactives) by their best docking score. Compute the **AUC-ROC** or **Enrichment Factor at 1%**. A conformation is good if actives rank high (strong predicted binding) and inactives rank low.

   `SAR_score(conf) = AUC( {dock_score(active_i, conf)}, {dock_score(inactive_j, conf)} )`

4. **Rank conformations** by SAR_score. The top-ranked conformations are those whose pocket geometry is maximally consistent with the SAR.

**Performance estimate:** GNINA takes ~1–5 seconds per ligand per pocket on a GPU (`--exhaustiveness 8`). For 20 actives + 80 inactives × 100 conformations × 2 pockets/conformation ≈ 20,000–100,000 docking runs ≈ 11–55 GPU-hours, tractable on a cluster.

#### Step 6: Final Protein-Ligand Relaxation (OpenMM)

After selecting the top-K (conformation, pose) pairs from the docking results, run a **local protein-ligand minimization** to produce physically relaxed structures suitable for downstream use.

Protocol:

1. **Build complex:** Combine the docked ligand pose from GNINA with the protein conformation.
2. **Define a local shell:** Select protein atoms within ~6–10 Å of any ligand atom as the flexible region.
3. **Restrain the rest of the system:** Apply positional restraints (e.g. harmonic, 5–20 kcal/mol/Å² equivalent) to protein atoms outside the shell so minimization remains local and does not distort global conformation.
4. **Energy minimize:** Use OpenMM (`LangevinIntegrator` + `Simulation.minimizeEnergy`) for short local relaxation (hundreds to a few thousand iterations).
5. **Quality checks:** Record minimized energy, ligand heavy-atom RMSD from the docked pose, and key contact preservation. Reject poses that drift excessively.

Because the docked poses from GNINA are already sterically valid and chemically reasonable, this minimization step is gentle refinement — not clash resolution — and is unlikely to produce the topological artifacts (ring-threading, atom interpenetration) that plagued the previous overlay→minimization approach.

#### Step 7: Validation & Comparison

Compute validation metrics on the top-K conformations:

- AUC-ROC for active/inactive discrimination (from GNINA docking scores)
- Enrichment factor at 1%, 5%, 10%
- Comparison against baselines: rigid docking into the input structure, ensemble docking into multiple PDB conformers (if available), and blind pocket detection (Fpocket/Pocketeer on the apo structure alone)

This validates whether the pipeline correctly identified SAR-consistent conformations.

---

### Bayesian Scoring Framework (Alternative / Extension)

An alternative to the enrichment-based score is a Bayesian formulation that provides a principled posterior probability over conformations:

**P(conformation | SAR data) ∝ P(actives bind well | conf) × P(inactives bind poorly | conf) × P(conf | force field)**

In practice, this decomposes as:

1. **Likelihood for actives — P(actives bind well | conf):** For each active compound *a*, the probability that it achieves a good fit score in this conformation. Modeled as a sigmoid over the fit score:

   `P(a binds | conf) = σ( (fit(a, conf) - θ) / τ )`

   where *θ* is a threshold fit score and *τ* controls sharpness. The joint likelihood over all actives is the product: `∏_a P(a binds | conf)`

2. **Likelihood for inactives — P(inactives bind poorly | conf):** The complementary probability — inactives should have *low* fit scores:

   `P(a_inact fails | conf) = 1 - σ( (fit(a_inact, conf) - θ) / τ )`

   Joint: `∏_j P(inactive_j fails | conf)`

3. **Prior — P(conf | force field):** The Boltzmann weight of the conformation from the MD simulation. Conformations that the protein visits frequently under the force field are more probable than rare, strained states. In practice, this is approximated by the cluster population (fraction of MD frames assigned to that cluster).

4. **Posterior:** The product of these three terms gives the posterior score. Conformations that are physically accessible (high prior), dock actives well (high active likelihood), and reject inactives (high inactive likelihood) score highest.

The advantage of the Bayesian formulation over raw enrichment is that it naturally incorporates the **thermodynamic plausibility** of each conformation — a rare, strained conformation that happens to discriminate actives from inactives is penalized relative to a thermodynamically stable one that achieves similar discrimination. It also provides a unified probabilistic framework that can be extended to incorporate additional evidence (e.g. mutagenesis data, known SAR cliffs) as additional likelihood terms.

---

### Additional Exploratory Methods & Computational Architectures

To bypass the computationally explosive bottleneck of running full atomic-level docking (or FEP) on thousands of transient MD frames, the framework can utilize intelligent surrogate loops and fast geometric filtering to evaluate the SAR constraints.

**1. Asynchronous "Fuzzy" Pharmacophore Matching** 

If more complex feature extraction is required, Geometric Deep Learning tools (like PharmacoNet) can be used to generate transient 3D pharmacophore graphs.

- The MD engine runs continuously, passing structurally distinct "macro-states" (triggered by a simple collective variable shift) to a parallel ML worker for graph matching against the SAR actives and inactives.

**2. Machine Learning Loops for Directed Sampling** To prevent the framework from blindly exploring un-druggable states, ML loops can actively guide the sampling toward SAR-compliant geometries.

- **Bayesian Optimization (BO):** Map the protein's flexibility into a low-dimensional latent space (via a VAE). Use a Gaussian Process to explore this space, guided by a discriminative objective function that rewards active binding and penalizes inactive binding
- **Active Learning Surrogate:** Train a fast 3D Graph Neural Network to predict the SAR-discrimination score directly from a given conformation. Only run expensive rigorous scoring (like FEP) on conformations where the GNN is highly uncertain, iteratively retraining the model to improve its accuracy.
- **Reinforcement Learning / Differentiable Biasing:** Convert the geometric overlap between the protein's alpha-spheres and the active ligand's pharmacophore into a continuous, differentiable function. This can be passed to an engine like PLUMED as a custom driving force, physically pushing the simulation to open an SAR-compliant pocket.

---

### Workflow Diagram: Alpha-Sphere + GNINA Pipeline

```mermaid
flowchart TB
    subgraph INPUTS["Inputs"]
        PDB["Protein Structure<br/>(PDB, apo or holo)"]
        ACTIVES["Active Compounds<br/>(SMILES + IC50)"]
        INACTIVES["Inactive Compounds<br/>(SMILES)"]
    end

    subgraph PREP["Ligand Preparation"]
        CONF_GEN["Generate 3D Conformers<br/>(RDKit ETKDG)"]
        SDF_OUT["Write SDF Files<br/>(GNINA input format)"]
    end

    subgraph MD_SAMPLING["Conformational Sampling"]
        MD["N Parallel MD Replicas<br/>(OpenMM, explicit solvent)"]
        FRAMES["Extract Frames<br/>(every 100 ps)"]
        CLUSTER["On-the-Fly Clustering<br/>(RMSD-based, cross-replica)"]
    end

    subgraph POCKET_DETECTION["Pocket Detection + Docking Box"]
        POCKETEER["Run Pocketeer<br/>(Alpha-Sphere Tessellation)"]
        DOCKBOX["Define Docking Boxes<br/>from Alpha-Sphere Cloud<br/>(centroid + extent + padding)"]
    end

    subgraph GNINA_DOCKING["GNINA Docking + SAR Scoring"]
        DOCK_ACT["Dock Actives<br/>(GNINA, CNN + Vina scoring)"]
        DOCK_INACT["Dock Inactives<br/>(GNINA, CNN + Vina scoring)"]
        SAR_SCORE["Compute SAR Discrimination<br/>AUC-ROC on CNN Scores"]
        RANK["Rank Conformations<br/>by SAR Discrimination"]
    end

    subgraph REFINEMENT["Final Relaxation"]
        TOP_CONF["Select Top-K<br/>(conformation, pose) pairs"]
        MINIMIZE["Local Protein-Ligand<br/>Minimization (OpenMM)"]
        QC["Quality Checks<br/>(energy, contacts, RMSD)"]
    end

    subgraph OUTPUTS["Outputs"]
        BEST_CONF["Best SAR-Consistent<br/>Protein Conformation"]
        POSES["Validated Binding Poses<br/>for Actives"]
        EXPLAIN["SAR Explanation<br/>(key contacts, pocket geometry)"]
        RECEPTOR["Optimized Receptor for<br/>Downstream Campaigns"]
    end

    ACTIVES --> CONF_GEN
    INACTIVES --> CONF_GEN
    CONF_GEN --> SDF_OUT

    PDB --> MD
    MD --> FRAMES
    FRAMES --> CLUSTER
    CLUSTER --> POCKETEER
    POCKETEER --> DOCKBOX

    SDF_OUT --> DOCK_ACT
    SDF_OUT --> DOCK_INACT
    DOCKBOX --> DOCK_ACT
    DOCKBOX --> DOCK_INACT
    DOCK_ACT --> SAR_SCORE
    DOCK_INACT --> SAR_SCORE
    SAR_SCORE --> RANK

    RANK --> TOP_CONF
    TOP_CONF --> MINIMIZE
    MINIMIZE --> QC

    QC --> BEST_CONF
    DOCK_ACT --> POSES
    SAR_SCORE --> EXPLAIN
    BEST_CONF --> RECEPTOR
```

### Deprecated Steps (retained for reference / interpretability)

The following components from the original pipeline have been deprecated for pose generation but may still be used for interpretability and SAR explanation:

- **Physicochemical Coloring of Alpha-Spheres** (`pocket_coloring.py`): Mapping protein environment features onto alpha-sphere clouds. Useful for explaining *why* a pocket discriminates actives from inactives, but no longer needed for the docking workflow.
- **Pharmacophore Overlay** (`overlay.py`): Rigid-body alignment of ligand pharmacophore clouds to alpha-sphere clouds. Replaced by GNINA docking which performs proper conformational search.
- **Local Minimization** (`local_minimization.py`): OpenMM minimization of overlaid poses. The overlay→minimize approach produced topological artifacts (ring-threading, atom interpenetration) because energy minimization cannot resolve topological violations. Now replaced by:
  1. GNINA docking (which naturally avoids such artifacts via Monte Carlo sampling)
  2. A final gentle minimization step applied only to physically valid docked poses

### Extended Vision: ML Feedback Loop (Future Work)

```mermaid
flowchart TB
    subgraph CORE["Core Pipeline - Paper 1"]
        direction TB
        MD["Parallel MD Sampling"] --> FRAMES["Frame Extraction"]
        FRAMES --> POCKETEER["Pocketeer Alpha-Spheres"]
        POCKETEER --> GEO_SCORE["Geometric SAR Scoring"]
        GEO_SCORE --> RANK["Rank Conformations"]
    end

    subgraph ML_LOOP["ML-Guided Sampling - Future"]
        direction TB
        VAE["VAE Latent Space<br/>(Protein Conformations)"]
        GP["Gaussian Process<br/>(BO Surrogate)"]
        GNN["3D-GNN Surrogate<br/>(Active Learning)"]
        PLUMED["PLUMED Differentiable<br/>Bias (RL)"]
    end

    subgraph RIGOR["Rigorous Scoring - Future"]
        direction TB
        FEP["FEP / RBFE"]
        GCMC["GCMC Occupancy"]
        SILCS["SILCS FragMaps"]
    end

    RANK -->|"top-K uncertain"| GNN
    RANK -->|"top-K confident"| FEP
    GNN -->|"retrain"| GP
    GP -->|"suggest next"| VAE
    VAE -->|"bias MD"| PLUMED
    PLUMED --> MD
```