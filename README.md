
# SpacGPA: Spatial and single-cell Gene Program Analysis

**SpacGPA** is a GPU-accelerated toolkit that annotates spatial transcriptomes through de novo interpretable gene programs. It builds co-expression networks via a **graphical Gaussian model**, identifies programs with a **modified Markov Clustering (MCL)** procedure, performs ontology-based enrichment (GO/MP), and applies programs to spatial analyses such as SVGs detection, spatial domain annotation, and label integration.

<p align="center">
  <img src="assets/SpacGPA_Workflow.png" alt="SpacGPA workflow" width="500" />
</p>

SpacGPA is described in the preprint “[SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs](https://doi.org/10.1101/2025.10.01.679918)” (Xu, Chen, Ma) and is released under the BSD-3-ClausW license.

## Highlights

- **Network inference:** Fast gene–gene co-expression via graphical Gaussian models with GPU-friendly blockwise ops.
- **Program discovery:** Interpretable gene programs from a modified **MCL** with minimal tuning.
- **Functional interpretation:** **GO/MP** enrichment and hub genes-weighted program scoring.
- **Spatial analysis:** SVGs detection, domain annotation, neighbor smoothing, and label integration.
- **Scale & transfer:** Handles million-spot datasets; programs transfer across sections, platforms, and single-cell data.

## Installation

### Option A — Create a conda environment (recommended)

1) Create the environment  
**conda env create -f environment.yml**  
**conda activate SpacGPA**  
2) Install SpacGPA (from the repository root)  
**cd The/root/directory/of/SpacGPA-main/you/downloaded**  
**pip install -e .**
**conda deactivated**  

**Attention：** If you prefer a different CUDA/toolkit, edit the PyTorch lines in environment.yml
to match your GPU / driver, or switch to CPU-only builds from pytorch. See [PyTorch](https://pytorch.org/get-started/locally/)
for a suitable Version build for your CUDA/CPU

### Option B — Planned binary releases

When published, installation will also be available via PyPI:  
**pip install spacgpa**  
or via conda-forge:  
**conda install -c conda-forge spacgpa**

## Usage

 • API and tutorials: See Documentation and Tutorials in [SpacGPA](https://spacgpa.readthedocs.io).  
 • Each tutorial covers a typical workflow: network construction → program identification (MCL) → functional enrichment analysis (GO/MP) → spatial transcriptome annotation using programs → visualization.

## How to cite

If you use SpacGPA in your work, please cite:
Xu Y, Chen L, Ma S. SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs. bioRxiv (2025). doi:10.1101/2025.10.01.679918.  ￼  ￼

## License

This project is released under the BSD-3-Clause license. See the LICENSE file for details.  ￼

## Contact

For questions and feature requests, please open an issue on GitHub or contact the corresponding author (<sma@ustc.edu.cn>).
