
# SpacGPA: Spatial and single-cell Gene Program Analysis

![SpacGPA logo](assets/SpacGPA_Logo.png)

SpacGPA is a GPU-accelerated toolkit that annotates spatial transcriptomes through de novo interpretable gene programs. It builds co-expression networks via a graphical Gaussian model, identifies programs with a modified Markov Clustering (MCL) procedure, performs ontology-based enrichment (GO/MP), and applies programs to spatial analyses such as SVG detection, spatial domain delineation, and label integration.

SpacGPA is described in the preprint “SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs” (Xu, Chen, Ma) and is released under the BSD-3-Clause license.

Highlights
 • Gene-centric GGM networks with iterative random sampling and block-matrix GPU operations (PyTorch) for speed and memory efficiency on very large datasets.  ￼
 • Program discovery using a modified, GPU-accelerated MCL (MCL-Hub) with minimal parameter tuning.  ￼
 • Functional interpretation: enrichment over ontologies (GO, MP) and hub-gene–weighted program scoring.  ￼
 • Spatial analytics: SVG identification (e.g., Moran’s I), Gaussian-mixture–based program-positive spot calling, spatial neighbor smoothing, and label integration strategies.  ￼
 • Scales to million-spot data and integrates across sections/platforms; programs transfer to independent spatial and single-cell datasets.  ￼  ￼

## Installation

Python: 3.8
OS: Linux/macOS/Windows (CUDA optional for GPU)
External binary: MCL (installed via conda as shown below)
PyTorch: install a build that matches your CUDA/CPU setup (see pytorch.org).

Option A — Create a conda environment (recommended)

1) Create the environment
conda env create -f environment.yml
conda activate spacgpa

2) (If you prefer a different CUDA/toolkit) edit the PyTorch lines in environment.yml
to match your GPU / driver, or switch to CPU-only builds from pytorch.

Note on MCL: the environment installs bioconda::mcl. If you use a custom environment, install it explicitly with
conda install -c bioconda mcl.

Option B — From source with pip

## system deps: ensure MCL is installed (e.g., conda install -c bioconda mcl)

## torch: install a suitable PyTorch build for your CUDA/CPU **before** installing spacgpa

## (see <https://pytorch.org/get-started/locally/>)

### install spacgpa

pip install -U pip setuptools wheel
pip install -e .          # from the repository root

Planned binary releases
When published, installation will also be available via:
 • PyPI: pip install spacgpa
 • conda-forge: conda install -c conda-forge spacgpa

## Usage

 • API and tutorials: See Documentation and Tutorials in XXX.
 • The documentation covers a typical workflow: network construction → program discovery (MCL) → enrichment (GO/MP) → SVG identification and spatial domain analysis → visualization utilities.

## Functionality at a glance

 • Network construction: partial correlations (graphical Gaussian model) with GPU-accelerated blockwise sampling.
 • Program discovery: modified MCL clustering; support for hub degree statistics.
 • Program interpretation: GO / MP enrichment; hub-gene–weighted program scores.
 • Spatial analyses:
 • SVG detection using degree and Moran’s I;
 • Gaussian-mixture–based program-positive calling for spatial domains;
 • Neighbor-aware smoothing;
 • Label integration combining expression-rank and spatial neighbors.
 • Visualization utilities:
 • Inter-program relationship heatmaps;
 • Program × cluster bubble plots;
 • Degree vs. Moran’s I scatter for program genes;
 • Program network diagrams;
 • Bar charts for GO/MP enrichment.

(Concepts and analyses summarized above are introduced in the preprint.  ￼)

## How to cite

If you use SpacGPA in your work, please cite:
Xu Y, Chen L, Ma S. SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs. bioRxiv (2025). doi:10.1101/2025.10.01.679918.  ￼  ￼

## License

This project is released under the BSD-3-Clause license. See the LICENSE file for details.  ￼

## Contact

For questions and feature requests, please open an issue on GitHub or contact the corresponding author (<sma@ustc.edu.cn>).
