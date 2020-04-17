import numpy as np

from autocti.util import fit_util


class FitDataset:

    # noinspection PyUnresolvedReferences
    def __init__(self, masked_dataset, model_data):
        """Class to fit a masked dataset where the dataset's data structures are any dimension.

        Parameters
        -----------
        masked_dataset : MaskedDataset
            The masked dataset (data, mask, noise map, etc.) that is fitted.
        model_data : ndarray
            The model data the masked dataset is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey data point.
        """
        self.masked_dataset = masked_dataset
        self.model_data = model_data

    @property
    def mask(self):
        return self.masked_dataset.mask

    @property
    def data(self):
        return self.masked_dataset.data

    @property
    def noise_map(self):
        return self.masked_dataset.noise_map

    @property
    def residual_map(self):
        """Compute the residual map between a masked dataset and model data, where:

        Residuals = (Data - Model_Data).

        The residual map values in masked pixels are returned as zero.
        """
        return fit_util.residual_map_from_data_mask_and_model_data(
            data=self.data, model_data=self.model_data, mask=self.mask
        )

    @property
    def normalized_residual_map(self):
        """Compute the normalized residual map between a masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise

        The normalized residual map values in masked pixels are returned as zero.
        """
        return fit_util.normalized_residual_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def chi_squared_map(self):
        """Computes the chi-squared map between a masked residual-map and noise map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

        The chi-squared map values in masked pixels are returned as zero.
        """
        return fit_util.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
        )

    @property
    def signal_to_noise_map(self):
        """The signal-to-noise_map of the dataset and noise map which are fitted."""
        signal_to_noise_map = np.divide(self.data, self.noise_map)
        signal_to_noise_map[np.isnan(signal_to_noise_map)] = 0
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map * np.invert(self.mask)

    @property
    def chi_squared(self):
        """Compute the chi-squared terms of each model data's fit to a masked dataset, by summing the masked
        chi-squared map of the fit.

        The chi-squared values in masked pixels are omitted from the calculation.
        """
        return fit_util.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=self.chi_squared_map, mask=self.mask
        )

    @property
    def reduced_chi_squared(self):
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))

    @property
    def noise_normalization(self):
        """Compute the noise map normalization terms of masked noise map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

        The noise map values in masked pixels are omitted from the calculation.
        """
        return fit_util.noise_normalization_from_noise_map_and_mask(
            noise_map=self.noise_map, mask=self.mask
        )

    @property
    def likelihood(self):
        return fit_util.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )


class FitImaging(FitDataset):
    def __init__(self, masked_imaging, model_image):
        """Class to fit a masked imaging dataset.

        Parameters
        -----------
        masked_imaging : MaskedImaging
            The masked imaging dataset that is fitted.
        model_image : Array
            The model image the masked imaging is fitted with.

        Attributes
        -----------
        residual_map : ndarray
            The residual map of the fit (data - model_data).
        chi_squared_map : ndarray
            The chi-squared map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared : float
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared : float
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of datas points), summed over \
            every data point.
        noise_normalization : float
            The overall normalization term of the noise_map, summed over every data point.
        likelihood : float
            The overall likelihood of the model's fit to the dataset, summed over evey data point.
        """

        super().__init__(masked_dataset=masked_imaging, model_data=model_image)

    @property
    def masked_imaging(self):
        return self.masked_dataset

    @property
    def image(self):
        return self.data

    @property
    def model_image(self):
        return self.model_data
