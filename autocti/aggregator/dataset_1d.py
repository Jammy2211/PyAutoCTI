from functools import partial
import autofit as af


def _dataset_1d_list_from(fit: af.Fit, use_dataset_full: bool = False):
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


class Dataset1DAgg:
    def __init__(self, aggregator: af.Aggregator, use_dataset_full: bool = False):
        self.aggregator = aggregator
        self.use_dataset_full = use_dataset_full

    def dataset_list_gen_from(
        self,
    ):
        """
        Returns a generator of `Dataset1D` objects from an input aggregator, which generates a list of the
        `Dataset1D` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `dataset_from_agg_obj` with the aggregator, which sets up each
        dataset_1d using only generators ensuring that manipulating the dataset_1d of large sets of results is done in
        a memory efficient way.

        Parameters
        ----------
        settings_dataset
            The settings of the `Dataset1D` object fitted by the non-linear search.
        """

        func = partial(_dataset_1d_list_from, use_dataset_full=self.use_dataset_full)
        return self.aggregator.map(func=func)
