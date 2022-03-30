.. _conda:

Installation with conda
=======================

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of `conda <https://conda.io/miniconda.html>`_.

First, create a conda environment (we name this ``autocti`` to signify it is for the **PyAutoCTI** install):

The command below creates this environment with some of the bigger package requirements, the rest will be installed
with **PyAutoFit** via pip:

.. code-block:: bash

    conda create -n autocti astropy numba numpy scikit-image scikit-learn scipy

Activate the conda environment (you will have to do this every time you want to run **PyAutoCTI**):

.. code-block:: bash

    conda activate autocti

Once you have created your conda environment you must install **arCTIc** before installing **PyAutoCTI**. The
installation guide is found at this link.

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoCTI** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies, and we assume ``numba`` and ``llvmlite`` were successfully installed when
creating the ``conda`` environment above,
see `here <https://pyautocti.readthedocs.io/en/latest/installation/troubleshooting.html>`_ for more details):

.. code-block:: bash

    pip install autocti==2022.03.18.2

You may get warnings which state something like:

.. code-block:: bash

    ERROR: autoarray 2022.2.14.1 has requirement numpy<=1.22.1, but you'll have numpy 1.22.2 which is incompatible.
    ERROR: numba 0.53.1 has requirement llvmlite<0.37,>=0.36.0rc1, but you'll have llvmlite 0.38.0 which is incompatible.

If you see these messages, they do not mean that the installation has failed and the instructions below will
identify clearly if the installation is a success.

Next, clone the ``autocti workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autocti_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autocti_workspace
   git clone https://github.com/Jammy2211/autocti_workspace --depth 1
   cd autocti_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py
