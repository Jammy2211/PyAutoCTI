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
    Returns a list of `FitDataset1D` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The masked dataset (e.g. data / noise map / pre cti data) as .fits files (contained in `dataset` folder).
    - The clocker used to add  CTI in the fit (`dataset/clocker.json`).
    - The settings used for clocking CIT (contained in `dataset/settings_cti.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a  list of `FitDataset1D` objects, by loading the masked
    dataset adding CTI to its pre-cti data via the cti model and clocking and fitting the model image to the dataset.

    If multiple `Dataset1D` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the datasets, perform the fit and return a list of `FitDataset1D` objects.

    If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
    to the database, the input `use_dataset_full` can be switched in to fit the full dataset instead.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    use_dataset_full
        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `Dataset1D` objects.
    clocker_list
        If input, overwrites the clocker used in the fit with a new clocker which is used to perform the fit.
    """

    from autocti.dataset_1d.fit import FitDataset1D

    dataset_list = _dataset_1d_list_from(fit=fit, use_dataset_full=use_dataset_full)

    if clocker_list is None:
        if not fit.children:
            clocker_list = [fit.value(name="clocker")]
        else:
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
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Dataset1D` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The masked dataset (e.g. data / noise map / pre cti data) as .fits files (contained in `dataset` folder).
        - The clocker used to add  CTI in the fit (`dataset/clocker.json`).
        - The settings used for clocking CIT (contained in `dataset/settings_cti.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create a `FitDataset1D` object via the `_fit_dataset_1d_from` method.

        This class's methods returns generators which create the instances of the `FitDataset1D` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Dataset1D` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Dataset1D` objects.

        If multiple `Dataset1D` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the datasets, perform the fit and return a list of `FitDataset1D` objects.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to fit the full dataset instead.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        use_dataset_full
            If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore
            accessible to the database, the input `use_dataset_full` can be switched in to load instead the
            full `Dataset1D` objects.
        clocker_list
            If input, overwrites the clocker used in the fit with a new clocker which is used to perform the fit.
        """
        super().__init__(
            aggregator=aggregator,
            use_dataset_full=use_dataset_full,
            clocker_list=clocker_list,
        )

    def object_via_gen_from(self, fit, cti: Union[CTI1D, CTI2D]) -> List[FitDataset1D]:
        """
        Returns a generator of `FitDataset1D` objects from an input aggregator.

        See `__init__` for a description of how the `FitDataset1D` objects are created by this method.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to fit the full dataset instead.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        cti
            The CTI model used to add CTI to the dataset to perform the fit.
        """
        return _fit_dataset_1d_list_from(
            fit=fit,
            cti=cti,
            use_dataset_full=self.use_dataset_full,
            clocker_list=self.clocker_list,
        )
