
# SpacGPA: Spatial and single-cell Gene Program Analysis

<img src="assets/SpacGPA_Logo.png" alt="SpacGPA logo" align="right" width="80" />

SpacGPA is a GPU-accelerated toolkit that annotates spatial transcriptomes through de novo interpretable gene programs. It builds co-expression networks via a graphical Gaussian model, identifies programs with a modified Markov Clustering (MCL) procedure, performs ontology-based enrichment (GO/MP), and applies programs to spatial analyses such as SVG detection, spatial domain delineation, and label integration.

<p align="center">
  <img src="assets/SpacGPA_Workflow.png" alt="SpacGPA workflow" width="600" />
</p>

SpacGPA is described in the preprint “[SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs](https://www.biorxiv.org/doi/10.1101/2025.10.01.679918)” (Xu, Chen, Ma) and is released under the BSD-3-ClausW license.

## Highlights

- **Network inference:** Fast gene–gene co-expression via graphical Gaussian models with GPU-friendly blockwise ops.
- **Program discovery:** Interpretable gene programs from a modified **MCL** with minimal tuning.
- **Functional interpretation:** **GO/MP** enrichment and hub genes-weighted program scoring.
- **Spatial analysis:** SVG detection (Moran’s *I*), Gaussian-mixture domain calling, neighbor smoothing, and label integration.
- **Scale & transfer:** Handles million-spot datasets; programs transfer across sections, platforms, and single-cell data.

## Installation

### Option A — Create a conda environment (recommended)

1) Create the environment  
**`conda env create -f environment.yml`**  
**`conda activate SpacGPA`**  
(Attention：If you prefer a different CUDA/toolkit, edit the PyTorch lines in environment.yml
to match your GPU / driver, or switch to CPU-only builds from pytorch. See [PyTorch]<https://pytorch.org/get-started/locally/>
for a suitable PyTorch Version build for your CUDA/CPU)  
2) Install SpacGPA (from the repository root)  
**`pip install -e .`**

### PyTorch

install a suitable PyTorch build for your CUDA/CPU **before** installing spacgpa

(see )

### Option B — Using

**`pip install -U pip setuptools wheel`**  
**`pip install -e .  # from the repository root`**

### Planned binary releases

When published, installation will also be available via:
**`PyPI: pip install spacgpa**`
or  
**`conda-forge: conda install -c conda-forge spacgpa**`

## Usage

 • API and tutorials: See Documentation and Tutorials in [SpacGPA](https://spacgpa.readthedocs.io).<br>
 • Each tutorial covers a typical workflow: network construction → program identification (MCL) → functional enrichment analysis (GO/MP) → spatial transcriptome annotation using programs → visualization.

## Functionality at a glance

### Network construction：

partial correlations (graphical Gaussian model) with GPU-accelerated blockwise sampling.

### Program discovery：

modified MCL clustering; support for hub degree statistics.

### Program interpretation：

GO / MP enrichment; hub-gene–weighted program scores.

### Spatial analyses：

SVGs detection via Gene's degree and Moran’s I;

Gaussian-mixture–based program-positive calling for spatial domains;

Label integration combining expression-rank and spatial neighbors.

### Visualization utilities:

Inter-program relationship heatmaps;

Program × cluster bubble plots;

Degree vs. Moran’s I scatter for program genes;

Program network diagrams;

Bar charts for GO/MP enrichment.

(Concepts and analyses summarized above are introduced in the preprint.  ￼)

## How to cite

If you use SpacGPA in your work, please cite:
Xu Y, Chen L, Ma S. SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs. bioRxiv (2025). doi:10.1101/2025.10.01.679918.  ￼  ￼

## License

This project is released under the BSD-3-Clause license. See the LICENSE file for details.  ￼

## Contact

For questions and feature requests, please open an issue on GitHub or contact the corresponding author (<sma@ustc.edu.cn>).
