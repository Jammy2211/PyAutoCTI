from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from autocti.clocker.abstract import AbstractClocker
    from autocti.model.model_util import CTI1D
    from autocti.model.model_util import CTI2D
    from autocti.charge_injection.fit import FitImagingCI

import autofit as af

from autocti.aggregator.abstract import AbstractAgg
from autocti.aggregator.imaging_ci import _imaging_ci_list_from


def _fit_imaging_ci_list_from(
    fit: af.Fit,
    cti: Union[CTI1D, CTI2D],
    use_dataset_full: bool = False,
    clocker_list: Optional[AbstractClocker] = None,
) -> List[FitImagingCI]:
    """
    Returns a `FitImagingCI` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    settings_dataset
        The settings of the `ImagingCI` object fitted by the non-linear search.

    Returns
    -------
    FitImagingCI
        The fit to the imaging_ci dataset computed via an instance of galaxies.
    """

    from autocti.charge_injection.fit import FitImagingCI

    imaging_ci_list = _imaging_ci_list_from(fit=fit, use_dataset_full=use_dataset_full)

    if clocker_list is None:
        clocker_list = [fit.value(name="clocker")]
        if clocker_list[0] is None:
            clocker_list = fit.child_values(name="clocker")

    post_cti_data_list = [
        clocker.add_cti(data=imaging_ci.data, cti=cti)
        for imaging_ci, clocker in zip(imaging_ci_list, clocker_list)
    ]

    return [
        FitImagingCI(
            dataset=imaging_ci,
            post_cti_data=post_cti_data,
        )
        for imaging_ci, post_cti_data in zip(imaging_ci_list, post_cti_data_list)
    ]


class FitImagingCIAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        use_dataset_full: bool = False,
        clocker_list: Optional[List[AbstractClocker]] = None,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to imaging_ci data, corresponding to the
        results of a non-linear search model-fit.

        Parameters
        ----------
        clocker
            The CTI arctic clocker used by aggregator's instances. If None is input, the clocker used by the
            non-linear search and model-fit is used.
        settings_dataset
            The settings of the `ImagingCI` object fitted by the non-linear search.
        """
        super().__init__(
            aggregator=aggregator,
            use_dataset_full=use_dataset_full,
            clocker_list=clocker_list,
        )

    def object_via_gen_from(self, fit, cti: Union[CTI1D, CTI2D]) -> FitImagingCI:
        """
        Creates a `FitImagingCI` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.

        Returns
        -------
        FitImagingCI
            A fit to imaging_ci data whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _fit_imaging_ci_list_from(
            fit=fit,
            cti=cti,
            use_dataset_full=self.use_dataset_full,
            clocker_list=self.clocker_list,
        )
