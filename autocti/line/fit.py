from autoarray.fit.fit_data import FitData
from autoarray.fit.fit_dataset import FitDataset

from autocti.line.dataset import DatasetLine


class FitDatasetLine(FitDataset):
    def __init__(self, dataset_line: DatasetLine, post_cti_data):
        """Fit a charge injection ci_data-set with a model cti image, also scalng the noises within a Bayesian
        framework.

        Parameters
        -----------
        dataset_line
            The charge injection image that is fitted.
        post_cti_data
            The `pre_cti_data` with cti added to it via the clocker and a CTI model.
        hyper_noise_scalars :
            The hyper_ci-parameter(s) which the noise_scaling_map_list_list is multiplied by to scale the noise-map.
        """

        fit = FitData(
            data=dataset_line.data,
            noise_map=dataset_line.noise_map,
            model_data=post_cti_data,
            mask=dataset_line.mask,
            use_mask_in_fit=True,
        )

        super().__init__(dataset=dataset_line, fit=fit)

    @property
    def dataset_line(self):
        return self.dataset

    @property
    def post_cti_data(self):
        return self.model_data

    @property
    def pre_cti_data(self):
        return self.dataset.pre_cti_data
