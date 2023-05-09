from functools import partial
from typing import Optional

import autofit as af
import autoarray as aa

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.dataset_1d.dataset_1d.settings import SettingsDataset1D


def _dataset_1d_from(fit: af.Fit, settings_dataset: Optional[SettingsDataset1D] = None):
    """
    Returns a `Dataset1D` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to create a `Dataset1D` generator.

     The `Dataset1D` is created following the same method as the PyAutoCTI `Search` classes, including using the
    `SettingsDataset1D` instance output by the Search to load inputs of the `Dataset1D`.

    Parameters
    ----------
    fit
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of model-fits.
    """

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    pre_cti_data = fit.value(name="pre_cti_data")
    layout = fit.value(name="layout")

    settings_dataset = settings_dataset or fit.value(name="settings_dataset")

    dataset_1d = Dataset1D(
        data=data,
        noise_map=noise_map,
        pre_cti_data=pre_cti_data,
        settings=settings_dataset,
        pad_for_convolver=True,
    )

    dataset_1d.apply_settings(settings=settings_dataset)

    return dataset_1d


class Dataset1DAgg:
    def __init__(self, aggregator: af.Aggregator):
        self.aggregator = aggregator

    def dataset_1d_gen_from(self, settings_dataset: Optional[aa.SettingsDataset1D] = None):
        """
        Returns a generator of `Dataset1D` objects from an input aggregator, which generates a list of the
        `Dataset1D` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `dataset_1d_from_agg_obj` with the aggregator, which sets up each
        dataset_1d using only generators ensuring that manipulating the dataset_1d of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        aggregator
            A PyAutoFit aggregator object containing the results of model-fits.
        """

        func = partial(_dataset_1d_from, settings_dataset=settings_dataset)

        return self.aggregator.map(func=func)
