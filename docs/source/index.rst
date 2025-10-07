SpacGPA: Spatial and single-cell Gene Program Analysis
=============================================================================

.. image:: _static/SpacGPA_Logo.png
   :alt: SpacGPA logo
   :align: center
   :width: 260px

.. rst-class:: lead

**SpacGPA** is a GPU-accelerated toolkit that annotates spatial transcriptomes through de 
novo interpretable gene programs. It builds co-expression networks via a **Gaussian graphical 
model (GGM)**, identifies programs with a **modified Markov Clustering (MCL)** algorithm, 
performs ontology-based enrichment (Gene Ontology (GO) / Mammalian Phenotype (MP)), and 
applies programs to spatial analyses such as detection of SVGs, spatial domain annotation, 
and label integration.

.. image:: _static/SpacGPA_Workflow.png
   :alt: SpacGPA workflow
   :align: center
   :width: 800px

This website hosts the installation guide, quick-start tutorial, in-depth
workflows, and full API reference.

.. admonition:: Recommended reading order
   :class: tip

   1. **Installation Guide**  
   2. **Quick Start**  
   3. **Tutorials** or the **API Reference** as needed.

.. raw:: html

   <hr/>

Clean Start
-----------

.. topic:: Quick Links

   - **Installation** → :doc:`installation`
   - **Quick Start** → :doc:`quickstart`
   - **Tutorials (workflows)** → :doc:`tutorials/index`
   - **API Reference** → :doc:`api/index`
   - **Changelog** → :doc:`changelog`

At a glance
-----------

- ✔️ **De novo interpretable gene programs** derived from GGM co-expression networks  
- ✔️ **Modified MCL** for robust program detection at scale  
- ✔️ **Ontology enrichment (GO / MP)** for functional interpretation  
- ✔️ **Spatial analyses**: SVG detection, domain annotation, label integration  
- ✔️ **GPU-accelerated** pipeline with reproducible Python APIs

Quick Installation
------------------

The full instructions are in :doc:`installation`. Below is the short version.

**From GitHub (latest):**
  
.. code-block:: bash

   pip install "git+https://github.com/MaShisongLab/SpacGPA.git"

**From source (development):**

.. code-block:: bash

   git clone https://github.com/MaShisongLab/SpacGPA.git
   cd SpacGPA
   pip install -e .

**Note.** GPU is optional but recommended for large datasets. See :doc:`installation` for environment details and troubleshooting.

--------------------------------------------------------
Table of Contents
--------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Quick Start
   :numbered:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

   changelog
   citation
   contributing
   license

--------------------------------------------------------
Citation & Acknowledgments
--------------------------------------------------------

If you use SpacGPA in your work, please cite:  
Xu Y, Chen L, Ma S. *SpacGPA: annotating spatial transcriptomes through de novo interpretable gene programs.* bioRxiv (2025). https://doi.org/10.1101/2025.10.01.679918

--------------------------------------------------------
Support & Contact
--------------------------------------------------------

* **GitHub**  : https://github.com/MaShisongLab/SpacGPA
* **Issues**  : Please open a ticket on GitHub Issues for bugs or feature requests
* **E-mail**  : sma@ustc.edu.cn