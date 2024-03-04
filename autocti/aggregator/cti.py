from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, Union

from autocti.aggregator.abstract import AggBase

if TYPE_CHECKING:
    from autocti.model.model_util import CTI1D
    from autocti.model.model_util import CTI2D

import autofit as af

logger = logging.getLogger(__name__)


def _cti_from(fit: af.Fit) -> Union[CTI1D, CTI2D]:
    """
    Returns a list of `CTI` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `CTI` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `CTI` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of Tracers. This is necessary if each Tracer has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.

    Returns
    -------
    CTI
        The list of CTI models computed via an instance of clocker.
    """

    if len(fit.children) > 0:
        logger.info(
            """
            Using database for a fit with multiple summed Analysis objects.

            CTI objects do not fully support this yet (e.g. adapt images may not be set up correctly)
            so proceed with caution!
            """
        )

    return fit.instance.cti


class CTIAgg(AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `CTI` objects from the results
    of a model-fit.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create an `CTI` object via the `_cti_from` method.

    This class's methods returns generators which create the instances of the `CTI` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `CTI` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `CTI` objects.

    If multiple `CTI` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of CTIs. This is necessary if each CTI has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(self, fit, instance: Optional[af.ModelInstance] = None) -> Union[CTI1D, CTI2D]:
        """
        Returns a generator of `CTI` objects from an input aggregator.

        See `__init__` for a description of how the `CTI` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).

        Returns
        -------
        CTI
            A list of `CTI` objects whose parameters are a sample of the non-linear search.
        """
        return _cti_from(fit=fit)
