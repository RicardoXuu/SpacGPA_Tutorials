
⸻

SpacGPA: Spatial and single-cell Gene Program Analysis

<img src="assets/SpacGPA_Logo.png" alt="SpacGPA logo" align="right" width="150" />


SpacGPA is a GPU-accelerated toolkit that annotates spatial transcriptomes through de novo interpretable gene programs. It builds co-expression networks via a graphical Gaussian model, identifies programs with a modified Markov Clustering (MCL) procedure, performs ontology-based enrichment (GO/MP), and applies programs to spatial analyses such as SVG detection, spatial domain delineation, and label integration.

SpacGPA is described in the preprint “SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs” (Xu, Chen, Ma) and is released under the BSD-3-Clause license.

⸻

Highlights
	•	Gene-centric GGM networks with iterative random sampling and block-matrix GPU operations (PyTorch) for speed and memory efficiency on very large datasets.  ￼
	•	Program discovery using a modified, GPU-accelerated MCL (MCL-Hub) with minimal parameter tuning.  ￼
	•	Functional interpretation: enrichment over ontologies (GO, MP) and hub-gene–weighted program scoring.  ￼
	•	Spatial analytics: SVG identification (e.g., Moran’s I), Gaussian-mixture–based program-positive spot calling, spatial neighbor smoothing, and label integration strategies.  ￼
	•	Scales to million-spot data and integrates across sections/platforms; programs transfer to independent spatial and single-cell datasets.  ￼  ￼

⸻

Installation

Python: 3.8
OS: Linux/macOS/Windows (CUDA optional for GPU)
External binary: MCL (installed via conda as shown below)
PyTorch: install a build that matches your CUDA/CPU setup (see pytorch.org).

Option A — Create a conda environment (recommended)

# 1) Create the environment
conda env create -f environment.yml
conda activate spacgpa

# 2) (If you prefer a different CUDA/toolkit) edit the PyTorch lines in environment.yml
#    to match your GPU / driver, or switch to CPU-only builds from pytorch.

Note on MCL: the environment installs bioconda::mcl. If you use a custom environment, install it explicitly with
conda install -c bioconda mcl.

Option B — From source with pip

# system deps: ensure MCL is installed (e.g., conda install -c bioconda mcl)

# torch: install a suitable PyTorch build for your CUDA/CPU **before** installing spacgpa
# (see https://pytorch.org/get-started/locally/)

# install spacgpa
pip install -U pip setuptools wheel
pip install -e .          # from the repository root

Planned binary releases
When published, installation will also be available via:
	•	PyPI: pip install spacgpa
	•	conda-forge: conda install -c conda-forge spacgpa

⸻

Usage
	•	API and tutorials: See Documentation and Tutorials in XXX.
	•	The documentation covers a typical workflow: network construction → program discovery (MCL) → enrichment (GO/MP) → SVG identification and spatial domain analysis → visualization utilities.

⸻

Functionality at a glance
	•	Network construction: partial correlations (graphical Gaussian model) with GPU-accelerated blockwise sampling.
	•	Program discovery: modified MCL clustering; support for hub degree statistics.
	•	Program interpretation: GO / MP enrichment; hub-gene–weighted program scores.
	•	Spatial analyses:
	•	SVG detection using degree and Moran’s I;
	•	Gaussian-mixture–based program-positive calling for spatial domains;
	•	Neighbor-aware smoothing;
	•	Label integration combining expression-rank and spatial neighbors.
	•	Visualization utilities:
	•	Inter-program relationship heatmaps;
	•	Program × cluster bubble plots;
	•	Degree vs. Moran’s I scatter for program genes;
	•	Program network diagrams;
	•	Bar charts for GO/MP enrichment.

(Concepts and analyses summarized above are introduced in the preprint.  ￼)

⸻

How to cite

If you use SpacGPA in your work, please cite:

Xu Y, Chen L, Ma S. SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs. bioRxiv (2025). doi:10.1101/2025.10.01.679918.  ￼  ￼

⸻

License

This project is released under the BSD-3-Clause license. See the LICENSE file for details.  ￼

⸻

Acknowledgments

We acknowledge the USTC Supercomputing Center and USTC School of Life Sciences Bioinformatics Center for computing resources.  ￼

⸻

Contact

For questions and feature requests, please open an issue on GitHub or contact the corresponding author (sma@ustc.edu.cn).  ￼

⸻

Notes for repository maintainers
	•	Place the logo image at assets/logo.png so the header graphic renders.
	•	Links labeled XXX should be replaced with the ReadTheDocs URLs once available.
	•	The environment.yml pins Python 3.8 and includes MCL. Adjust PyTorch CUDA version lines to your infrastructure.

⸻

Draft packaging files

These are minimal, working starters aligned to the dependencies you specified. You can commit them as-is and iterate.

setup.py

from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="spacgpa",                      # package name on PyPI
    version="0.1.0",
    description="Spatial and single-cell Gene Program Analysis (SpacGPA)",
    long_description=README,
    long_description_content_type="text/markdown",
    author="MaShisong Lab",
    url="https://github.com/MaShisongLab/SpacGPA",
    license="BSD-3-Clause",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    python_requires=">=3.8,<3.9",
    install_requires=[
        # graph / clustering
        "igraph>=0.10",          # PyPI name for python-igraph
        "leidenalg>=0.10",
        "python-louvain>=0.16",
        "louvain>=0.8.1",        # optional; safe to keep if used internally
        "networkx>=2.8",
        # bio + I/O
        "scanpy>=1.9",
        "mygene>=3.2",
        "reportlab>=4.0",
        "tables>=3.8",           # PyPI name (conda uses 'pytables')
        # typical scientific stack pulled by scanpy, but listed explicitly for clarity
        "numpy>=1.22",
        "scipy>=1.8",
        "pandas>=1.4",
        "anndata>=0.9",
        "matplotlib>=3.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # entry_points can be added here if you expose console scripts
)

Why no torch in install_requires? Users should install the PyTorch build that matches their hardware/driver stack; we avoid forcing a specific wheel.

⸻

environment.yml

name: spacgpa
channels:
  - conda-forge
  - bioconda
  - pytorch
  - nvidia
dependencies:
  - python=3.8
  - pip
  # graph stack
  - python-igraph>=0.10
  - leidenalg>=0.10
  - louvain>=0.8.1
  - python-louvain>=0.16
  - networkx>=2.8
  # bio / analysis
  - scanpy>=1.9
  - anndata>=0.9
  - numpy>=1.22
  - scipy>=1.8
  - pandas>=1.4
  - matplotlib>=3.5
  - mygene>=3.2
  - reportlab>=4.0
  - pytables>=3.8        # conda package name for PyTables
  - mcl                  # from bioconda (Markov Clustering binary)
  # GPU (edit these lines to match YOUR CUDA/driver)
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8    # for CUDA 11.8 drivers (Linux)
  - cudatoolkit          # optional; pin to your system if needed
  - pip:
      # if some deps are only on PyPI in your environment, list them here
      # - spacgpa          # uncomment after you publish to PyPI
      - tables>=3.8       # mirrors pip package name (optional redundancy)

If your machine has a different CUDA version, replace pytorch-cuda=11.8 with the appropriate build (or remove it and use CPU-only PyTorch).

⸻

Publishing checklists

PyPI (source + wheels)
	1.	Version & metadata
	•	Update version in setup.py.
	•	Ensure README.md, LICENSE, and (optional) MANIFEST.in are present.
	2.	Build & check

python -m pip install --upgrade pip build twine
python -m build
twine check dist/*


	3.	Upload

twine upload dist/*

(If the name spacgpa is taken on PyPI, consider spacgpa-tools.)

	4.	Post-publish
	•	Tag the release on GitHub.
	•	Add installation snippets (pip install spacgpa) to this README and docs.

conda-forge
	1.	Wait for (or point to) a PyPI sdist/wheel as the upstream source.
	2.	Fork conda-forge/staged-recipes and add a recipes/spacgpa/meta.yaml like:

{% set name = "spacgpa" %}
{% set version = "0.1.0" %}

package:
  name: {{ name }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/s/spacgpa/spacgpa-{{ version }}.tar.gz
  sha256: <fill after building>

build:
  noarch: python
  script: {{ PYTHON }} -m pip install . -vv

requirements:
  host:
    - python >=3.8,<3.9
    - pip
  run:
    - python >=3.8,<3.9
    - python-igraph
    - leidenalg
    - louvain
    - python-louvain
    - networkx
    - scanpy
    - mygene
    - reportlab
    - pytables
    - matplotlib
    - numpy
    - scipy
    - pandas
    # torch intentionally omitted; users install per hardware
    - mcl  # optional: provide a run constraint; users can also install separately

about:
  home: https://github.com/MaShisongLab/SpacGPA
  summary: Spatial and single-cell Gene Program Analysis
  license: BSD-3-Clause
  license_file: LICENSE

test:
  imports:
    - spacgpa


	3.	Open a PR; address bot feedback (missing hashes, dependency availability, etc.).
	4.	After the feedstock is created, update this README with conda-forge badges and conda install instructions.

⸻

Sources
	•	Title/authors/abstract and DOI for the SpacGPA preprint.  ￼  ￼  ￼
	•	Software availability and framework overview.  ￼  ￼
	•	Repository license is BSD-3-Clause (GitHub repo metadata).  ￼

If you’d like, I can also generate a small API index from the codebase once the package layout stabilizes (modules, key functions, and suggested docstrings).