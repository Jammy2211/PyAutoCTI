from functools import partial

import autofit as af
import autoarray as aa

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D


def _dataset_1d_list_from(fit: af.Fit, use_dataset_full: bool = False):
    """
    Returns a `Dataset1D` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The 1D dataset data as a .fits file (`dataset/data.fits`).
    - The noise-map as a .fits file (`dataset/noise_map.fits`).
    - The pre CTI data as a .fits file (`dataset/pre_cti_data.fits`).
    - The layout of the `Dataset1D` data structure used in the fit (`dataset/layout.json`).
    - The mask used to mask the `Dataset1D` data structure in the fit (`dataset/mask.fits`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Dataset1D` object, has the mask applied to the
    `Dataset1D` data structure and its settings updated to the values used by the model-fit.

    If multiple `Dataset1D` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, pre CTI data, layout and mask and combine them into a list of
    `Dataset1D` objects.

    If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
    to the database, the input `use_dataset_full` can be switched in to load instead the full `Dataset1D` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    use_dataset_full
        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `Dataset1D` objects.
    """

    # TODO : Rich make a really nice API for making this concise...

    if use_dataset_full:
        folder = "dataset_full"
    else:
        folder = "dataset"

    if fit.value(name=f"{folder}.data") is not None:
        layout = fit.value(name=f"{folder}.layout")

        data = aa.Array1D.from_primary_hdu(primary_hdu=fit.value(name=f"{folder}.data"))
        noise_map = aa.Array1D.from_primary_hdu(
            primary_hdu=fit.value(name=f"{folder}.noise_map")
        )
        pre_cti_data = aa.Array1D.from_primary_hdu(
            primary_hdu=fit.value(name=f"{folder}.pre_cti_data")
        )

        mask = aa.Mask1D.from_primary_hdu(primary_hdu=fit.value(name=f"{folder}.mask"))

        dataset = Dataset1D(
            data=data,
            noise_map=noise_map,
            pre_cti_data=pre_cti_data,
            layout=layout,
        )

        dataset = dataset.apply_mask(mask=mask)

        return dataset

    dataset_list = []

    for layout, data, noise_map, pre_cti_data, mask in zip(
        fit.child_values(name=f"{folder}.layout"),
        fit.child_values(name=f"{folder}.data"),
        fit.child_values(name=f"{folder}.noise_map"),
        fit.child_values(name=f"{folder}.pre_cti_data"),
        fit.child_values(name=f"{folder}.mask"),
    ):
        data = aa.Array1D.from_primary_hdu(primary_hdu=data)
        noise_map = aa.Array1D.from_primary_hdu(primary_hdu=noise_map)
        pre_cti_data = aa.Array1D.from_primary_hdu(primary_hdu=pre_cti_data)

        mask = aa.Mask1D.from_primary_hdu(primary_hdu=mask)

        dataset = Dataset1D(
            data=data,
            noise_map=noise_map,
            pre_cti_data=pre_cti_data,
            layout=layout,
        )

        dataset = dataset.apply_mask(mask=mask)

        dataset_list.append(dataset)

    return dataset_list


class Dataset1DAgg:
    def __init__(self, aggregator: af.Aggregator, use_dataset_full: bool = False):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Dataset1D` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The 1D dataset data as a .fits file (`dataset/data.fits`).
        - The noise-map as a .fits file (`dataset/noise_map.fits`).
        - The pre CTI data as a .fits file (`dataset/pre_cti_data.fits`).
        - The layout of the `Dataset1D` data structure used in the fit (`dataset/layout.json`).
        - The mask used to mask the `Dataset1D` data structure in the fit (`dataset/mask.fits`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `Dataset1D` object via the `_dataset_1d_from` method.

        This class's methods returns generators which create the instances of the `Dataset1D` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Dataset1D` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Dataset1D` objects.

        If multiple `Dataset1D` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, pre CTI data, layout and mask and combine them into a list of
        `Dataset1D` objects.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `Dataset1D` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        use_dataset_full
            If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore
            accessible to the database, the input `use_dataset_full` can be switched in to load instead the
            full `Dataset1D` objects.
        """
        self.aggregator = aggregator
        self.use_dataset_full = use_dataset_full

    def dataset_list_gen_from(
        self,
    ):
        """
        Returns a generator of `Dataset1D` objects from an input aggregator.

        See `__init__` for a description of how the `Dataset1D` objects are created by this method.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `Dataset1D` objects.
        """

        func = partial(_dataset_1d_list_from, use_dataset_full=self.use_dataset_full)
        return self.aggregator.map(func=func)
