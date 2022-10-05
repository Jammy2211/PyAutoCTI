.. _arctic:

Installing arCTIc
=================

The c++ package **arCTIc**, or the **AlgoRithm for Charge Transfer Inefficiency (CTI) Correction**, must be installed
alongside **PyAutoCTI**.

**arCTIc** models the CCD read out process, including the effects of CTI. **PyAutoCTI** wraps the c++ **arCTIc** source code
via a Cython wrapper, making it straight forward to import and use in **PyAutoCTI**.

This page gives the installation instructions for **arCTIc**, which are taken from its GitHub page (https://github.com/jkeger/arctic).

[A long term goal is for **arCTIc** to be made pip installable when one installs **PyAutoCTI** via pip. This requires clever wrapping of Cython libraries,
which non of the lead developers currently know how to do. If you think you could help us out please contact us!]


MacOS / Linux
-------------

First, clone the arctic GitHub repository (the ``sudo`` is only required for MacOS and not linux):

.. code-block:: bash

    git clone https://github.com/jkeger/arctic.git
    cd arctic
    sudo make all

If you are running Python in conda, add arctic to conda:

[NOTE: Certain versions of conda use the command ``conda develop`` (without a dash) instead of those shown below.]

.. code-block:: bash

    conda-develop arctic

If you are not using conda, add arctic to your ``PYTHONPATH`` environment variable:

.. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:/path/to/arctic

Also add arctic to your ``DYLD_LIBRARY_PATH`` environment variable:

.. code-block:: bash

    export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/path/to/arctic

You should now get output from running the following Python code in an Python interpreter:

.. code-block:: bash

    import numpy as np
    import arcticpytest

    arcticpytest.add_cti(np.zeros((5,5)))

**arCTIc** is now installed and you can install **PyAutoCTI** via conda or pip.