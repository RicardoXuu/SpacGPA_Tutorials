Installation
============

Option A — Quick Installation
-----------------------------

1. **Download the repository**

   - Clone:
     
     .. code-block:: bash

        git clone https://github.com/MaShisongLab/SpacGPA.git
        cd SpacGPA

   - Or download ZIP (GitHub → **Code** → **Download ZIP**) and unzip:

     .. code-block:: bash

        cd SpacGPA-main

2. **Make sure you have installed CUDA-enabled PyTorch for GPU support**  
   (skip this step if using CPU only).

3. **Install SpacGPA**

   .. code-block:: bash

      pip install .

Option B — Create a conda environment
-------------------------------------

1. **Create the environment**

   .. code-block:: bash

      # If you downloaded ZIP
      cd SpacGPA-main
      # If you cloned
      # cd SpacGPA
      conda env create -f environment.yml
      conda activate SpacGPA

2. **Install SpacGPA**

   .. code-block:: bash

      pip install .
      conda deactivate

.. note::

   If you prefer a different CUDA/toolkit, edit the PyTorch lines in
   ``environment.yml`` to match your GPU / driver before installation.
   See PyTorch for a suitable version build for your CUDA/CPU.
   Option B provides a conservative working environment; try replacing
   the software in ``environment.yml`` with a newer version if you expect
   faster computations.