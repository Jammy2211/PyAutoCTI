========
Modeling
========

Analysis
========

The ``Analysis`` objects define the ``log_likelihood_function`` of how a galaxy model is fitted to a dataset.

It acts as an interface between the data, model and the non-linear search.

.. currentmodule:: autocti

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   AnalysisImagingCI
   AnalysisDataset1D

.. currentmodule:: autocti

Settings
--------

Input into an ``Analysis`` class to customize the behaviour of a CTI model-fit performed via a non-linear search.

.. autosummary::
   :toctree: generated/

    SettingsCTI1D
    SettingsCTI2D

Non-linear Searches
-------------------

A non-linear search is an algorithm which fits a model to data.

**PyAutoGalaxy** currently supports three types of non-linear search algorithms: nested samplers,
Markov Chain Monte Carlo (MCMC) and optimizers.

.. currentmodule:: autofit

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   DynestyStatic
   DynestyDynamic
   Emcee
   PySwarmsLocal
   PySwarmsGlobal

Priors
------

The priors of parameters of every component of a mdoel, which is fitted to data, are customized using ``Prior`` objects.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   UniformPrior
   GaussianPrior
   LogUniformPrior
   LogGaussianPrior