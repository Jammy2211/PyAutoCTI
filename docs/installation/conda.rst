.. _conda:

Installation with conda
=======================

Install
-------

Installation via a conda environment circumvents compatibility issues when installing certain libraries. This guide
assumes you have a working installation of `conda <https://conda.io/miniconda.html>`_.

First, update conda:

.. code-block:: bash

    conda update -n base -c defaults conda

First, create a conda environment (we name this ``autocti`` to signify it is for the **PyAutoCTI** install):

The command below creates this environment with some of the bigger package requirements, the rest will be installed
via pip:

.. code-block:: bash

    conda create -n autocti astropy numpy scikit-image scikit-learn scipy

Activate the conda environment (you will have to do this every time you want to run **PyAutoCTI**):

.. code-block:: bash

    conda activate autocti

We upgrade pip to ensure certain libraries install:

.. code-block:: bash

    pip install --upgrade pip

The latest version of **PyAutoCTI** is installed via pip as follows (specifying the version as shown below ensures
the installation has clean dependencies):

.. code-block:: bash

    pip install autocti

You may get warnings which state something like:

.. code-block:: bash

    ERROR: autoarray 2023.7.7.2 has requirement numpy<=1.22.1, but you'll have numpy 1.22.2 which is incompatible.
    ERROR: numba 0.53.1 has requirement llvmlite<0.37,>=0.36.0rc1, but you'll have llvmlite 0.38.0 which is incompatible.

If you see these messages, they do not mean that the installation has failed and the instructions below will
identify clearly if the installation is a success.

If there is an error (e.g. issues with installing GSL) check out the `troubleshooting section <https://pyautocti.readthedocs.io/en/latest/installation/troubleshooting.html>`_.

Numba
-----

Numba (https://numba.pydata.org)  is an optional library which makes **PyAutoCTI** run a lot faster, which we
strongly recommend users have installed.

You can install numba (versions above 0.56.4 are not supported) via the following command:

.. code-block:: bash

    pip install numba==0.56.4

Some users have experienced difficulties installing numba, which is why it is an optional library. If your
installation is not successful, you can use **PyAutoCTI** without it installed for now, to familiarize yourself
with the software and determine if it is the right software for you.

If you decide that **PyAutoCTI** is the right software, then I recommend you commit the time to getting a
successful numba install working, with more information provided `at this readthedocs page <https://pyautocti.readthedocs.io/en/latest/installation/numba.html>`_

Workspace
---------

Next, clone the ``autocti workspace`` (the line ``--depth 1`` clones only the most recent branch on
the ``autocti_workspace``, reducing the download size):

.. code-block:: bash

   cd /path/on/your/computer/you/want/to/put/the/autocti_workspace
   git clone https://github.com/Jammy2211/autocti_workspace --depth 1
   cd autocti_workspace

Run the ``welcome.py`` script to get started!

.. code-block:: bash

   python3 welcome.py

It should be clear that **PyAutoCTI** runs without issue.

If there is an error check out the `troubleshooting section <https://pyautocti.readthedocs.io/en/latest/installation/troubleshooting.html>`_.