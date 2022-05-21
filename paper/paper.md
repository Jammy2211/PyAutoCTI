---
title: "`PyAutoCTI`: Open-Source Multiwavelength Galaxy Structure & Morphology"
tags:
  - astronomy
  - instrumentation
  - Charge Coupled Devices
  - Python
  - weak lensing
authors:
  - name: James. W. Nightingale
    orcid: 0000-0002-8987-7401
    affiliation: 1
  - name: Richard J. Massey
    orcid: 0000-0002-6085-3780
    affiliation: 1 
  - name: Jacob Kegerreis
    orcid: 0000-0001-5383-236X
    affiliation: 2 
  - name: Richard G. Hayes
    affiliation: 1
affiliations:
  - name: Institute for Computational Cosmology, Stockton Rd, Durham DH1 3LE
    index: 1

date: 12 May 2022
codeRepository: https://github.com/Jammy2211/PyAutoCTI
license: MIT
bibliography: paper.bib
---

# Summary

Over the past half a century, space telescopes have given us an extraordinary view of the Universe, for example detecting 
the faint spectra of the first galaxies in the Universe [@Dunlop2013] [@Bouwens2015], measuring tiny distortions in
galaxy shapes due to gravitational lensing [@Massey2007] [@Schrabback2010] and a high precision map of stars within
the Milky Way [@Brown2018] [@Brown2020]. These observations require a deep understanding of the telescope's
instrumental characteristics, including the calibration and correction of charge transfer inefficiency (hereafter CTI) [@Massey2010d] [@Massey2014],
a phenomena where radiation damage to the telescope's charge-coupled device (CCD) sensors leads to gradually
increased smearing in acquired exposures over the telescope's lifetime.

`PyAutoCTI` is an open-source Python 3.6+ package for the calibration of CTI for space telescopes. By interfacing
with `arCTIc` (`the algorithm for Charge Transfer Inefficiency correction`) the calibrated CTI models can straightforwardly
be used to correct and remove CTI in every science image taken throughout the telescope's lifetime. Core features 
include fully automated Bayesian model-fitting of CTI calibration data, support for different calibration
datasets (e.g. charge injection imaging, warm pixels) and a database for building a temporal model of CTI
over the telescope's lifetime. The software places a focus on big data analysis, including support for 
graphical models that simultaneously fits large CTI calibration datasets and an SQLite3 database that allows extensive 
suites of calibration results to be loaded, queried and analysed. Accompanying `PyAutoCTI` is 
the [autocti workspace](https://github.com/Jammy2211/autocti_workspace), which includes example scripts, datasets and an
overview of core `PyAutoCTI` functionality. Readers can  try `PyAutoCTI` right now by going 
to [the introduction Jupyter notebook on Binder](https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/release) or 
checkout the [readthedocs](https://pyautocti.readthedocs.io/en/latest/) for a complete overview of `PyAutoCTI`'s 
features.

# Background

During an exposure, photons hitting a CCD generate photo-electrons that are held in discrete clouds (pixels) by a grid 
of electrostatic potentials. At the end of the exposure, the potentials are varied so as to move clouds of electrons 
first in the 'parallel' direction to the edge of the CCD, then along the perpendicular 'serial' register to an 
amplifier and read-out electronics that count the number of electrons. The CCD substrate through which the electrons 
move is built of a silicon lattice. However, defects in the lattice known as 'traps' can temporarily capture electrons. 
If this time is longer than the time taken to move a cloud of electrons to the next pixel, an electron may not be 
released back into its original charge cloud, but to a subsequent one: creating a characteristic 'trailing' or 'smearing' 
effect behind sources in the image. 

![An example of CTI calibration performed using `PyAutoCTI`. The figure shows CTI calibration data which includes cosmic rays (left panel); the CTI model data inferred by fitting this data (middle panel) and the data corrected for CTI using this model (right panel) .\label{figure:example1}](hstcombined.png)

Figure 1 shows an example CTI calibration dataset which is fitted using `PyAutoCTI`. The left panel shows simulated
charge injection line imaging calibration data, where a row of special pixels have electronically injected a known 
number of electrons into the CCD before clocking. CTI trails are seen in the parallel clocking direction (downwards)
and serial clocking direction (rightwards). The charge injection shows some degree of non-uniformity and the data
includes cosmic rays, systematic effects `PyAutoCTI` accounts for. The middle panel of
Figure 1 shows a `PyAutoCTI` fit to this data with a CTI model, which accurately reproduces the trails seen in the
left panel. In the right panel this CTI model is used to accurately correct and remove CTI from the data; this 
correction would be applied to science imaging. Charge injection data is the primary form of CTI calibration data 
anticipated to be used for the European Space Agency's Euclid space mission, however `PyAutoCTI` also supports other 
datasets such as warm pixels which were used to calibrate CTI for the Hubble Space Telescope [@Massey2010d] .
 
# Statement of Need

Space based telescopes planned for launch over the coming decades will map out the Universe's dark matter via
weak gravitational lensing, make precision detections of exoplanets that are further from Earth than ever
seen before [@Halverson2016] and ?. For these ambitious projects to be successful they all require CTI is removed
from science imaging with very stringent requirements, neccesitating that CTI calibration provides in-depth knowledge 
about CTI on every CCD. `PyAutoCTI` ensures that large CTI calibration datasets can be exploited to measure the CTI model 
over the telescope's life and it provides tools which maximally extract information from these datasets using contemporary 
Bayesian inference techniques.

# Software API and Features

At the heart of the `PyAutoCTI` API are `Trap` objects, which represent the populations of traps on a CCD which cause
CTI. The volume filling behaviour of a CCD is modeled via `CCD` objects, which is combined with traps to compose CTI 
models which add CTI to a mock CTI calibration data via `arCTIc`. `PyAutoCTI` has dedicated objects for specific
CTI calibration datasets, for example the `ImagingCI` object for charge injection data, alongside bespoke tools
for visualizing these datasets and masking them before fitting.  The `astropy` cosmology module 
handles unit conversions and calculations are optimized using the packages `NumPy` [@numpy] and `numba` [@numba].

To perform model-fitting, `PyAutoCTI` adopts the probabilistic programming  
language `PyAutoFit` (https://github.com/rhayes777/PyAutoFit). `PyAutoFit` allows users to compose a 
model from `Trap` and `CCD` objects, customize the model parameterization and fit it to data via a 
non-linear search, for example `dynesty` [@dynesty], `emcee` [@emcee] or `PySwarms` [@pyswarms]. `PyAutoFit`'s 
graphical modeling framework allows one to fit a temporal model to a suite of CTI calibration data. Using a technique 
called expectation propagation [@Vehtari2020], the framework fits each dataset one-by-one and combines the results of 
every fit into a temporal model using a self-consistent Bayesian framework. To ensure the analysis and interpretation of 
fits to large datasets is feasible, `PyAutoFit`'s database tools write modeling results to a relational database which 
can be queried from a storage drive to a Python script or Jupyter notebook. This uses memory-light `Python` generators, 
ensuring it is practical for use over a telescope's entire lifetime, where CTI calibration data taken with a daily
cadence may consist of thousands of datasets.

# Workspace

`PyAutoCTI` is distributed with the [autocti workspace](https://github.com/Jammy2211/autocti_workspace>), which 
contains example scripts for modeling and simulating CTI. The workspace is accessible 
on [Binder](https://mybinder.org/v2/gh/Jammy2211/autocti_workspace/HEAD) and example scripts can therefore be run 
without a local `PyAutoCTI` installation.

# Software Citations

`PyAutoCTI` is written in Python 3.6+ [@python] and uses the following software packages:

- `Astropy` [@astropy1] [@astropy2]
- `corner.py` [@corner]
- `dynesty` [@dynesty]
- `emcee` [@emcee]
- `Matplotlib` [@matplotlib]
- `numba` [@numba]
- `NumPy` [@numpy]
- `PyAutoFit` [@pyautofit]
- `pyprojroot` (https://github.com/chendaniely/pyprojroot)
- `PySwarms` [@pyswarms]
- `scikit-image` [@scikit-image]
- `scikit-learn` [@scikit-learn]
- `Scipy` [@scipy]

# Related Software

- `arCTIc` https://github.com/jkeger/arctic [@Massey2014]

# Acknowledgements

JWN and RJM are supported by the UK Space Agency, through grant ST/V001582/1, and by InnovateUK through grant TS/V002856/1. 
RGH is supported by STFC Opportunities grant ST/T002565/1.
# AA, QH, CSF and SMC are supported by ERC Advanced In-vestigator grant, DMIDAS [GA 786910] and also by the STFCConsolidated 
# Grant for Astronomy at Durham [grant numbersST/F001166/1, ST/I00162X/1,ST/P000541/1].
RJM is supported by a Royal Society University Research Fellowship.
This work used the DiRAC@Durham facility managed by the Institute for Computational Cosmology on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The equipment was funded by BEIS capital funding via STFC capital grants ST/K00042X/1, ST/P002293/1, ST/R002371/1 and ST/S002502/1, Durham University and STFC operations grant ST/R000832/1. DiRAC is part of the National e-Infrastructure.

# References
