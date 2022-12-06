.. _arctic:

Installing arCTIc
=================

The c++ package **arCTIc**, or the **AlgoRithm for Charge Transfer Inefficiency (CTI) Correction**, must be installed
alongside **PyAutoCTI**.

**arCTIc** models the CCD read out process, including the effects of CTI. **PyAutoCTI** wraps the c++ **arCTIc** source code
via a Cython wrapper, making it straight forward to import and use in **PyAutoCTI**.

**arCTIc** is a requirement of **PyAutoCTI** and should be installed when the following command is run:

.. code-block:: bash

    pip install autocti

You can check that **arCTIc** has been installed via the command:

.. code-block:: bash

    pip show arcticpy

You should get the following text:

.. code-block:: bash

    Name: arcticpy
    Version: 2.1
    Summary: This is the python module for the arCTIc code
    Home-page:
    Author:
    Author-email: Richard Massey <r.j.massey@durham.ac.uk>
    License:
    Location: /mnt/c/Users/Jammy/Code/PyAuto/arctic/python
    Requires: autoconf, numpy
    Required-by:

If an error is raised when trying to install **arCTIc** you should first try to install it via `pip`:

.. code-block:: bash

    pip install arcticpy

If the installation does not work, you can follow additional installation instructions for **arCTIc** on its GitHub page (https://github.com/jkeger/arctic).
