=========
CTI Model
=========

---
CTI
---

Package all of the components of a CTI model together (e.g. ``Traps``s, ``CCD``s) for passing to ``arCTIc``.

.. currentmodule:: autocti

.. autosummary::
   :toctree: generated/

    CTI1D
    CTI2D

-----
Traps
-----

The traps on a CCD which cause CTI.

Different ``Trap`` objects make different assumptions for how they capture and release electrons.

.. autosummary::
   :toctree: generated/

    TrapInstantCapture
    TrapSlowCapture
    TrapInstantCaptureContinuum

---
CCD
---

The volume-filling behaviour of a CCD, describing how an electron cloud fills a pixel.

This what traps electron capturing is subject to as a function of pixel normalization.

.. autosummary::
   :toctree: generated/

    CCDPhase

