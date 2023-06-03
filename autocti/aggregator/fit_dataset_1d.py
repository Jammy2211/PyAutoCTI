from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from autocti.clocker.abstract import AbstractClocker
    from autocti.model.model_util import CTI1D
    from autocti.model.model_util import CTI2D
    from autocti.dataset_1d.fit import FitDataset1D

import autofit as af

from autocti.aggregator.abstract import AbstractAgg
from autocti.aggregator.dataset_1d import _dataset_1d_list_from


def _fit_dataset_1d_list_from(
    fit: af.Fit,
    cti: Union[CTI1D, CTI2D],
    use_dataset_full: bool = False,
    clocker_list: Optional[AbstractClocker] = None,
) -> List[FitDataset1D]:
    """
    Returns a `FitDataset1D` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    use_dataset_full
        If True, the full dataset is used to create the dataset list, else the masked dataset is used. This is used
        when trying to plot regions of the dataset that are masked out (e.g. the FPR).
    settings_dataset
        The settings of the `Dataset1D` object fitted by the non-linear search.

    Returns
    -------
    FitDataset1D
        The fit to the dataset_1d dataset computed via an instance of galaxies.
    """

    from autocti.dataset_1d.fit import FitDataset1D

    dataset_list = _dataset_1d_list_from(fit=fit, use_dataset_full=use_dataset_full)

    if clocker_list is None:
        clocker_list = [fit.value(name="clocker")]
        if clocker_list[0] is None:
            clocker_list = fit.child_values(name="clocker")

    post_cti_data_list = [
        clocker.add_cti(data=dataset.pre_cti_data, cti=cti)
        for dataset, clocker in zip(dataset_list, clocker_list)
    ]

    return [
        FitDataset1D(
            dataset=dataset,
            post_cti_data=post_cti_data,
        )
        for dataset, post_cti_data in zip(dataset_list, post_cti_data_list)
    ]


class FitDataset1DAgg(AbstractAgg):
    def __init__(
        self,
        aggregator: af.Aggregator,
        use_dataset_full: bool = False,
        clocker_list: Optional[List[AbstractClocker]] = None,
    ):
        """
        Wraps a PyAutoFit aggregator in order to create generators of fits to dataset_1d data, corresponding to the
        results of a non-linear search model-fit.

        Parameters
        ----------
        use_dataset_full
            If True, the full dataset is used to create the dataset list, else the masked dataset is used. This is used
            when trying to plot regions of the dataset that are masked out (e.g. the FPR).
        clocker_list
            The CTI arctic clocker used by aggregator's instances. If None is input, the clocker used by the
            non-linear search and model-fit is used.
        """
        super().__init__(
            aggregator=aggregator,
            use_dataset_full=use_dataset_full,
            clocker_list=clocker_list,
        )

    def object_via_gen_from(self, fit, cti: Union[CTI1D, CTI2D]) -> FitDataset1D:
        """
        Creates a `FitDataset1D` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.

        Returns
        -------
        FitDataset1D
            A fit to dataset_1d data whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _fit_dataset_1d_list_from(
            fit=fit,
            cti=cti,
            use_dataset_full=self.use_dataset_full,
            clocker_list=self.clocker_list,
        )
