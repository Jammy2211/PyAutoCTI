from functools import partial

import autofit as af


def _imaging_ci_list_from(fit: af.Fit, use_dataset_full: bool = False):
    """
    Returns a `ImagingCI` object from an aggregator's `SearchOutput` class, which we call an 'agg_obj' to describe
    that it acts as the aggregator object for one result in the `Aggregator`. This uses the aggregator's generator
    outputs such that the function can use the `Aggregator`'s map function to create a `ImagingCI` generator.

     The `ImagingCI` is created following the same method as the PyAutoCTI `Search` classes, including using the
    `SettingsImagingCI` instance output by the Search to load inputs of the `ImagingCI`.

    Parameters
    ----------
    fit
        A PyAutoFit aggregator's SearchOutput object containing the generators of the results of model-fits.
    use_dataset_full
        If True, the full dataset is used to create the dataset list, else the masked dataset is used. This is used
        when trying to plot regions of the dataset that are masked out (e.g. the FPR).
    """

    name = "dataset"

    if use_dataset_full:
        name = "dataset_full"

    dataset_list = [fit.value(name=name)]
    if dataset_list[0] is None:
        dataset_list = fit.child_values(name=name)

    return dataset_list


class ImagingCIAgg:
    def __init__(self, aggregator: af.Aggregator, use_dataset_full: bool = False):
        self.aggregator = aggregator
        self.use_dataset_full = use_dataset_full

    def dataset_list_gen_from(
        self,
    ):
        """
        Returns a generator of `ImagingCI` objects from an input aggregator, which generates a list of the
        `ImagingCI` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `dataset_from_agg_obj` with the aggregator, which sets up each
        imaging_ci using only generators ensuring that manipulating the imaging_ci of large sets of results is done in
        a memory efficient way.

        Parameters
        ----------
        settings_dataset
            The settings of the `ImagingCI` object fitted by the non-linear search.
        """

        func = partial(_imaging_ci_list_from, use_dataset_full=self.use_dataset_full)
        return self.aggregator.map(func=func)
