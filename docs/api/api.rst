=============
API Reference
=============

---------------
Data Structures
---------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Mask1D
   Mask2D
   Array1D
   Array2D
   ValuesIrregular
   Grid1D
   Grid2D
   Grid2DIterate
   Grid2DIrregular

----------
Dataset 1D
----------

.. currentmodule:: autocti

.. autosummary::
   :toctree: generated/

   Dataset1D
   SettingsDataset1D
   SimulatorDataset1D
   Layout1D
   FitDataset1D
   AnalysisDataset1D

------------------------
Charge Injection Imaging
------------------------

.. currentmodule:: autocti

.. autosummary::
   :toctree: generated/

   ImagingCI
   SettingsImagingCI
   SimulatorImagingCI
   Layout2DCI
   FitImagingCI
   AnalysisImagingCI

-----
Traps
-----

.. currentmodule:: autocti

.. autosummary::
   :toctree: generated/

    TrapInstantCapture
    TrapSlowCapture
    TrapInstantCaptureContinuum

---
CCD
---

    CCDPhase

--------------------------
Read Out Electronics (ROE)
--------------------------

    ROE
    ROEChargeInjection

--------
Clocking
--------

    Clocker1D
    Clocker2D
    SimulatorCosmicRayMap

------------
CTI Modeling
------------

.. currentmodule:: autocti

**Setup:**

.. autosummary::
   :toctree: generated/

    CTI1D
    CTI2D
    SettingsCTI1D
    SettingsCTI2D

**Searches:**

.. currentmodule:: autofit

.. autosummary::
   :toctree: generated/

   DynestyStatic
   DynestyDynamic
   Emcee
   PySwarmsLocal
   PySwarmsGlobal

-----
Plots
-----

.. currentmodule:: autocti.plot

**Plotters:**

.. autosummary::
   :toctree: generated/

    Array2DPlotter
    Array1DPlotter
    YX1DPlotter
    Dataset1DPlotter
    FitDataset1DPlotter
    ImagingCIPlotter
    FitImagingCIPlotter

**Search Plotters:**

.. autosummary::
   :toctree: generated/

   DynestyPlotter
   UltraNestPlotter
   EmceePlotter
   ZeusPlotter
   PySwarmsPlotter

**Plot Customization Objects**

.. autosummary::
   :toctree: generated/

    MatPlot1D
    MatPlot2D
    Include1D
    Include2D
    Visuals1D
    Visuals2D

**Matplotlib Wrapper Base Objects:**

.. autosummary::
   :toctree: generated/

    Axis
    Units
    Figure
    Cmap
    Colorbar
    ColorbarTickParams
    TickParams
    YTicks
    XTicks
    Title
    YLabel
    XLabel
    Legend
    Output

**Matplotlib Wrapper 1D Objects:**

.. autosummary::
   :toctree: generated/

    YXPlot

**Matplotlib Wrapper 2D Objects:**

.. autosummary::
   :toctree: generated/

    ArrayOverlay
    GridScatter
    GridPlot
    VectorYXQuiver
    PatchOverlay
    VoronoiDrawer
    OriginScatter
    MaskScatter
    BorderScatter
    PositionsScatter
    IndexScatter
    MeshGridScatter
    ParallelOverscanPlot
    SerialPrescanPlot
    SerialOverscanPlot