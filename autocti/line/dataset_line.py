from autoarray.mask import mask_1d
from autoarray.structures.arrays.one_d import array_1d
from autoarray.dataset import abstract_dataset


class SettingsDatasetLine(abstract_dataset.AbstractSettingsDataset):

    pass


class DatasetLine(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        data: array_1d.Array1D,
        noise_map: array_1d.Array1D,
        pre_cti_line: array_1d.Array1D,
        settings: SettingsDatasetLine = SettingsDatasetLine(),
    ):

        super().__init__(data=data, noise_map=noise_map, settings=settings)

        self.data = data
        self.noise_map = noise_map
        self.pre_cti_line = pre_cti_line

    def apply_mask(self, mask: mask_1d.Mask1D) -> "DatasetLine":

        data = array_1d.Array1D.manual_mask(array=self.data, mask=mask).native
        noise_map = array_1d.Array1D.manual_mask(array=self.noise_map, mask=mask).native

        return DatasetLine(
            data=data, noise_map=noise_map, pre_cti_line=self.pre_cti_line
        )

    def apply_settings(self, settings: SettingsDatasetLine) -> "DatasetLine":

        return self
