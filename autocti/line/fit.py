import autoarray as aa

from autocti.line.dataset import DatasetLine


class FitDatasetLine(aa.FitDataset):
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

        super().__init__(dataset=dataset_line)

        self.post_cti_data = post_cti_data

    @property
    def dataset_line(self) -> DatasetLine:
        return self.dataset

    @property
    def mask(self) -> aa.Mask1D:
        return self.dataset.mask

    @property
    def model_data(self) -> aa.Array1D:
        return self.post_cti_data

    @property
    def pre_cti_data(self) -> aa.Array1D:
        return self.dataset.pre_cti_data
