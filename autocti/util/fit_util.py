import numpy as np


def residual_map_from_data_mask_and_model_data(data, mask, model_data):
    """Compute the residual map between a masked dataset and model data, where:

    Residuals = (Data - Model_Data).

    Parameters
    -----------
    data : np.ndarray
        The observed data that is fitted.
    mask : np.ndarray
        The mask applied to the dataset, where *False* entries are included in the calculation.
    model_data : np.ndarray
        The model data used to fit the observed data.
    """
    return np.subtract(
        data, model_data, out=np.zeros_like(data), where=np.asarray(mask) == 0
    )


def normalized_residual_map_from_residual_map_noise_map_and_mask(
    residual_map, noise_map, mask
):
    """Compute the normalized residual map between a masked dataset and model data, where:

    Normalized_Residual = (Data - Model_Data) / Noise

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the observed dataset.
    noise_map : np.ndarray
        The noise map of the observed dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.divide(
        residual_map,
        noise_map,
        out=np.zeros_like(residual_map),
        where=np.asarray(mask) == 0,
    )


def chi_squared_map_from_residual_map_noise_map_and_mask(residual_map, noise_map, mask):
    """Computes the chi-squared map between a masked residual-map and noise map, where:

    Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)

    Parameters
    -----------
    residual_map : np.ndarray
        The residual-map of the model-simulator fit to the observed dataset.
    noise_map : np.ndarray
        The noise map of the observed dataset.
    mask : np.ndarray
        The mask applied to the residual-map, where *False* entries are included in the calculation.
    """
    return np.square(
        np.divide(
            residual_map,
            noise_map,
            out=np.zeros_like(residual_map),
            where=np.asarray(mask) == 0,
        )
    )


def chi_squared_from_chi_squared_map_and_mask(chi_squared_map, mask):
    """Compute the chi-squared terms of each model data's fit to an observed dataset, by summing the masked
    chi-squared map of the fit.

    Parameters
    ----------
    chi_squared_map : np.ndarray
        The chi-squared map of values of the model-simulator fit to the observed dataset.
    mask : np.ndarray
        The mask applied to the chi-squared map, where *False* entries are included in the calculation.
    """
    return np.sum(chi_squared_map[np.asarray(mask) == 0])


def noise_normalization_from_noise_map_and_mask(noise_map, mask):
    """Compute the noise map normalization terms of masked noise map, summing the noise_map value in every pixel as:

    [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

    Parameters
    ----------
    noise_map : np.ndarray
        The masked noise map of the observed dataset.
    mask : np.ndarray
        The mask applied to the noise map, where *False* entries are included in the calculation.
    """
    return np.sum(np.log(2 * np.pi * noise_map[np.asarray(mask) == 0] ** 2.0))


def likelihood_from_chi_squared_and_noise_normalization(
    chi_squared, noise_normalization
):
    """Compute the log likelihood of each masked 1D model-simulator fit to the dataset, where:

    Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)

    Parameters
    ----------
    chi_squared : float
        The chi-squared term for the model-simulator fit to the observed dataset.
    noise_normalization : float
        The normalization noise_map-term for the observed dataset's noise map.
    """
    return -0.5 * (chi_squared + noise_normalization)
