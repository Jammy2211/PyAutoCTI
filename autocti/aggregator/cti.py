from __future__ import annotations
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from autocti.model.model_util import CTI1D
    from autocti.model.model_util import CTI2D

import autofit as af

from autocti.aggregator.abstract import AbstractAgg


def _cti_from(fit: af.Fit) -> Union[CTI1D, CTI2D]:
    """
    Returns a `CTI` object from a PyAutoFit database `Fit` object and an instance of a CTI object from a non-linear
    search model-fit.

    This function adds the `adapt_model_image` and `adapt_cti_image_path_dict` to the clocker before constructing
    the `CTI`, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    clocker
        The CTI arctic clocker used by the non-linear search and model-fit.

    Returns
    -------
    CTI
        The cti computed via an instance of clocker.
    """

    return fit.instance


class CTIAgg(AbstractAgg):
    """
    Wraps a PyAutoFit aggregator in order to create generators of ctis corresponding to the results of a non-linear
    search model-fit.
    """

    def object_via_gen_from(self, fit, cti: Union[CTI1D, CTI2D]) -> Union[CTI1D, CTI2D]:
        """
        Creates a `CTI` object from a `ModelInstance` that contains the clocker of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.
        cti
            A CTI model corresponding to a sample of the non-linear search.

        Returns
        -------
        CTI
            A cti whose clocker are a sample of a PyAutoFit non-linear search.
        """
        return _cti_from(fit=fit)
