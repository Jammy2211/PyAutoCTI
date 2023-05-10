from functools import partial
from typing import Optional

import autofit as af

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
    settings_dataset
        The settings of the `Dataset1D` object fitted by the non-linear search.
    """

    dataset = fit.value(name="dataset")

    settings_dataset = settings_dataset or fit.value(name="settings_dataset")

    dataset.apply_settings(settings=settings_dataset)

    return dataset


class Dataset1DAgg:
    def __init__(self, aggregator: af.Aggregator):
        self.aggregator = aggregator

    def dataset_gen_from(self, settings_dataset: Optional[SettingsDataset1D] = None):
        """
        Returns a generator of `Dataset1D` objects from an input aggregator, which generates a list of the
        `Dataset1D` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `dataset_from_agg_obj` with the aggregator, which sets up each
        dataset_1d using only generators ensuring that manipulating the dataset_1d of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        settings_dataset
            The settings of the `Dataset1D` object fitted by the non-linear search.
        """

        func = partial(_dataset_1d_from, settings_dataset=settings_dataset)

        return self.aggregator.map(func=func)
