from functools import partial

import autoarray as aa
import autofit as af

from autocti.charge_injection.imaging.imaging import ImagingCI


def _imaging_ci_list_from(fit: af.Fit, use_dataset_full: bool = False):
    """
    Returns a `ImagingCI` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The charge injection dataset data as a .fits file (`dataset/data.fits`).
    - The noise-map as a .fits file (`dataset/noise_map.fits`).
    - The pre CTI data as a .fits file (`dataset/pre_cti_data.fits`).
    - The cosmic ray map (if included) as a .fits file (`dataset/cosmic_ray_map.fits`).
    - The layout of the `ImagingCI` data structure used in the fit (`dataset/layout.json`).
    - The settings dictionary of the data (if used) as a .json file (`dataset/settings_dict.json`).
    - The mask used to mask the `ImagingCI` data structure in the fit (`dataset/mask.fits`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `ImagingCI` object, has the mask applied to the
    `ImagingCI` data structure and its settings updated to the values used by the model-fit.

    If multiple `ImagingCI` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, pre CTI data, layout and mask and combine them into a list of
    `ImagingCI` objects.

    If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
    to the database, the input `use_dataset_full` can be switched in to load instead the full `ImagingCI` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    use_dataset_full
        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `ImagingCI` objects.
    """

    if use_dataset_full:
        folder = "dataset_full"
    else:
        folder = "dataset"

    if not fit.children:
        fit_list = [fit]
    else:
        fit_list = fit.children

    dataset_list = []

    for fit in fit_list:
        layout = fit.value(name=f"{folder}.layout")

        data = aa.Array2D.from_primary_hdu(primary_hdu=fit.value(name=f"{folder}.data"))
        noise_map = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name=f"{folder}.noise_map")
        )
        pre_cti_data = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name=f"{folder}.pre_cti_data")
        )
        cosmic_ray_map = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name=f"{folder}.cosmic_ray_map")
        )

        settings_dict = fit.value(name="dataset.settings_dict")

        dataset = ImagingCI(
            data=data,
            noise_map=noise_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            settings_dict=settings_dict,
            layout=layout,
        )

        mask = aa.Mask2D.from_primary_hdu(primary_hdu=fit.value(name=f"{folder}.mask"))

        dataset_list.append(dataset.apply_mask(mask=mask))

    return dataset_list


class ImagingCIAgg:
    def __init__(self, aggregator: af.Aggregator, use_dataset_full: bool = False):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `ImagingCI` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The charge injection dataset data as a .fits file (`dataset/data.fits`).
        - The noise-map as a .fits file (`dataset/noise_map.fits`).
        - The pre CTI data as a .fits file (`dataset/pre_cti_data.fits`).
        - The cosmic ray map (if included) as a .fits file (`dataset/cosmic_ray_map.fits`).
        - The layout of the `ImagingCI` data structure used in the fit (`dataset/layout.json`).
        - The settings dictionary of the data (if used) as a .json file (`dataset/settings_dict.json`).
        - The mask used to mask the `ImagingCI` data structure in the fit (`dataset/mask.fits`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `ImagingCI` object via the `_dataset_1d_from` method.

        This class's methods returns generators which create the instances of the `ImagingCI` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `ImagingCI` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `ImagingCI` objects.

        If multiple `ImagingCI` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
        method is instead used to load lists of the data, noise-map, pre CTI data, layout and mask and combine them
        into a list of `ImagingCI` objects.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `ImagingCI` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        use_dataset_full
            If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore
            accessible to the database, the input `use_dataset_full` can be switched in to load instead the
            full `ImagingCI` objects.
        """
        self.aggregator = aggregator
        self.use_dataset_full = use_dataset_full

    def dataset_list_gen_from(
        self,
    ):
        """
        Returns a generator of `ImagingCI` objects from an input aggregator.

        See `__init__` for a description of how the `ImagingCI` objects are created by this method.

        If a `dataset_full` is input into the `Analysis` class when a model-fit is performed and therefore accessible
        to the database, the input `use_dataset_full` can be switched in to load instead the full `ImagingCI` objects.
        """
        func = partial(_imaging_ci_list_from, use_dataset_full=self.use_dataset_full)
        return self.aggregator.map(func=func)
