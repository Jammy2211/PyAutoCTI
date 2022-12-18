import copy
from typing import Dict, Optional


import autoarray as aa

from autocti.preloads import Preloads
from autocti.charge_injection.imaging.imaging import ImagingCI


class FitImagingCI(aa.FitImaging):
    def __init__(
        self,
        dataset: ImagingCI,
        post_cti_data: aa.Array2D,
        hyper_noise_scalar_dict: Optional[Dict] = None,
        preloads: Preloads = Preloads(),
    ):
        """
        Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        ----------
        dataset
            The charge injection image that is fitted.
        post_cti_data
            The `pre_cti_data` with cti added to it via the clocker and a CTI model.
        hyper_noise_scalar_dict
            The hyper_ci-parameter(s) which the noise_scaling_map_dict_list is multiplied by to scale the noise-map.
        """

        super().__init__(dataset=dataset, use_mask_in_fit=True)

        self.post_cti_data = post_cti_data
        self.hyper_noise_scalar_dict = hyper_noise_scalar_dict
        self.noise_scaling_map_dict = dataset.noise_scaling_map_dict
        self.preloads = preloads

    @property
    def imaging_ci(self) -> ImagingCI:
        return self.dataset

    @property
    def noise_map(self) -> aa.Array2D:

        if self.hyper_noise_scalar_dict is not None:

            return hyper_noise_map_from(
                hyper_noise_scalar_dict=self.hyper_noise_scalar_dict,
                noise_scaling_map_dict=self.noise_scaling_map_dict,
                noise_map=self.dataset.noise_map,
            )

        return self.dataset.noise_map

    @property
    def model_data(self) -> aa.Array2D:
        return self.post_cti_data

    @property
    def layout(self):
        return self.imaging_ci.layout

    @property
    def pre_cti_data(self):
        return self.dataset.pre_cti_data

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.

        If the dataset includes a noise covariance matrix, this is used instead to account for covariance in the
        goodness-of-fit.

        The standard chi-squared calculation in PyAutoArray computes the `chi-squared` from the `residual_map`
        and `chi_squared_map`, which requires that the `ndarrays` which are used to do this are created and stored
        in memory. For charge injection imaging, the large datasets mean this can be computationally slow.

        This function computes the `chi_squared` directly from the data, avoiding the need to store the data in memory
        and offering faster tune times.
        """

        return aa.util.fit.chi_squared_with_mask_fast_from(
            data=self.dataset.data,
            noise_map=self.noise_map,
            mask=self.mask,
            model_data=self.model_data
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """

     #   if self.preloads.noise_normalization is not None:
     #       return self.preloads.noise_normalization

        return aa.util.fit.noise_normalization_with_mask_from(
            noise_map=self.noise_map, mask=self.mask
        )

    @property
    def chi_squared_map_of_regions_ci(self):
        return self.layout.extract.regions_array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_parallel_eper(self):
        return self.layout.extract.parallel_eper.array_2d_from(
            array=self.chi_squared_map
        )

    @property
    def chi_squared_map_of_serial_eper(self):
        return self.layout.extract.serial_eper.array_2d_from(array=self.chi_squared_map)

    @property
    def chi_squared_map_of_serial_overscan_no_eper(self):
        return self.layout.extract.serial_overscan_above_eper_array_2d_from(
            array=self.chi_squared_map
        )


def hyper_noise_map_from(hyper_noise_scalar_dict, noise_scaling_map_dict, noise_map):
    """
    For a noise-map, use the model hyper noise and noise-scaling maps to compute a scaled noise-map.

    Parameters
    -----------
    hyper_noise_scalar_dict
    noise_scaling_map_dict
    noise_map : imaging.NoiseMap or ndarray
        An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
        second.
    """

    noise_map = copy.copy(noise_map)

    for key, hyper_noise_scalar in hyper_noise_scalar_dict.items():

        noise_map += hyper_noise_scalar.scaled_noise_map_from(
            noise_scaling=noise_scaling_map_dict[key]
        )

    return noise_map
