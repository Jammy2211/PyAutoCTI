===============
Data Structures
===============

2D Data Structures
------------------

Two-dimensional data structures store and mask 2D arrays containing data (e.g. images) and
grids of (y,x) Cartesian coordinates.

.. currentmodule:: autocti

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Mask2D
   Array2D
   Grid2D
   Grid2DIterate
   Grid2DIrregular

Charge Injection Imaging
------------------------

For charge injection datasets taken with a CCD (or similar imaging device).

.. autosummary::
   :toctree: generated/

   ImagingCI
   Layout2DCI
   SettingsImagingCI
   SimulatorImagingCI

1D Data Structures
------------------

One-dimensional data structures store and mask 1D arrays containing data (e.g. 1D CTI datasets
images) and grids of (y,x) Cartesian coordinates.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Mask1D
   Array1D
   ValuesIrregular
   Grid1D

Dataset 1D
----------

For 1D datasets.

.. autosummary::
   :toctree: generated/

   Dataset1D
   LayoutDataset1D
   SettingsDataset1D
   SimulatorDataset1D