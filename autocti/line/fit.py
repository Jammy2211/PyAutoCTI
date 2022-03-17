import autoarray as aa

from autocti.line.dataset import DatasetLine


class FitDatasetLine(aa.FitDataset):
    def __init__(self, dataset: DatasetLine, post_cti_data):
        """
        Fit a 1D CTI dataset with model cti data.

        Parameters
        -----------
        dataset
            The charge injection image that is fitted.
        post_cti_data
            The `pre_cti_data` with cti added to it via the clocker and a CTI model.
        """

        super().__init__(dataset=dataset)

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
