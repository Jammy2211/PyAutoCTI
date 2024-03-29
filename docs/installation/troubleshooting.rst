.. _troubleshooting:

Troubleshooting
===============

GSL
---

`arcticpy` depends on the GSL libraries, which can cause installation issues.

If your installation of **PyAutoCTI** fails due to GSL, try installing the following version of arcticpy which does
not depend on GSL, after you have installed ``autocti``:

.. code-block:: bash

    pip install arcticpy_no_gsl

Current Working Directory
-------------------------

**PyAutoCTI** scripts assume that the ``autocti_workspace`` directory is the Python working directory. This means
that, when you run an example script, you should run it from the ``autocti_workspace`` as follows:

.. code-block:: bash

    cd path/to/autocti_workspace (if you are not already in the autocti_workspace).
    python3 scripts/dataset_fit/modeling/start_here.py

The reasons for this are so that **PyAutoCTI** can:

 - Load configuration settings from config files in the ``autocti_workspace/config`` folder.
 - Load example data from the ``autocti_workspace/dataset`` folder.
 - Output the results of models fits to your hard-disk to the ``autocti/output`` folder.
 - Import modules from the ``autocti_workspace``, for example ``from autocti_workspace.transdimensional import pipelines``.

If you have any errors relating to importing modules, loading data or outputting results it is likely because you
are not running the script with the ``autocti_workspace`` as the working directory!

Matplotlib Backend
------------------

Matplotlib uses the default backend on your computer, as set in the config file:

.. code-block:: bash

    autocti_workspace/config/visualize/general.ini

If unchanged, the backend is set to 'default', meaning it will use the backend automatically set up for Python on
your system.

.. code-block:: bash

    general:
      backend: default

There have been reports that using the default backend causes crashes when running the test script below (either the
code crashes without a error or your computer restarts). If this happens, change the config's backend until the test
works (TKAgg has worked on Linux machines, Qt5Agg has worked on new MACs). For example:

.. code-block:: bash

    general:
      backend: TKAgg

Support
-------

If you are still having issues with installation or using **PyAutoCTI** in general, please raise an issue on the
`autocti_workspace issues page <https://github.com/Jammy2211/autocti_workspace/issues>`_ with a description of the
problem and your system setup (operating system, Python version, etc.).