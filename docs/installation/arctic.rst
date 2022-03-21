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


