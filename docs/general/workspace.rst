.. _workspace:

Workspace Tour
==============

You should have downloaded and configured the `autocti workspace <https://github.com/Jammy2211/autocti_workspace>`_
when you installed **PyAutoCTI**.

If you didn't, checkout the
`installation instructions <https://pyautocti.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
for how to downloaded and configure the workspace.

The ``README.rst`` files distributed throughout the workspace describe every folder and file, and specify if
examples are for beginner or advanced users.

New users should begin by checking out the following parts of the workspace.

Scripts / Notebooks
-------------------

There are numerous example describing how to perform ctiing calculations, cti modeling, and many other
**PyAutoCTI** features. All examples are provided as Python scripts and Jupyter notebooks.

Descriptions of every configuration file and their input parameters are provided in the ``README.rst`` in
the `config directory of the workspace <https://github.com/Jammy2211/autocti_workspace/tree/release/config>`_

Config
------

Here, you'll find the configuration files which customize:

    - The default settings used by every non-linear search.
    - Visualization, including the backend used by *matplotlib*.
    - The priors and notation configs associated with the light and mass profiles used for cti modeling.
    - The behaviour of different (y,x) Cartesian grids used to perform cti calculations.
    - The general.ini config which customizes other aspects of **PyAutoCTI**.

Checkout the `configuration <https://pyautocti.readthedocs.io/en/latest/general/installation.html#installation-with-pip>`_
section of the readthedocs for a complete description of every configuration file.

Dataset
-------

Contains the dataset's used to perform cti modeling. Example datasets using simulators included with the workspace
are included here by default.

Output
------

The folder where cti modeling results are stored.

SLaM
----

Advanced cti modeling pipelines that use the Source, Light and Mass (SLaM) approach to cti modeling.

See `here <https://pyautocti.readthedocs.io/en/latest/advanced/slam.html>`_ for an overview.