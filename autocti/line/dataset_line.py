import numpy as np
from typing import Union, Optional, List, Tuple

from autoarray.mask import mask_1d
from autoarray.dataset import abstract_dataset

from arcticpy.src.ccd import CCDPhase
from arcticpy.src.traps import AbstractTrap
from autoarray.structures.arrays.one_d.array_1d import array_1d_util
from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.dataset import preprocess
from autoarray.dataset.imaging import AbstractSimulatorImaging
from autocti.line.layout_line import Layout1DLine
from autocti.util.clocker import Clocker1D
from autocti import exc


class SettingsDatasetLine(abstract_dataset.AbstractSettingsDataset):

    pass


class DatasetLine(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        data: Array1D,
        noise_map: Array1D,
        pre_cti_line: Array1D,
        layout: Layout1DLine,
        settings: SettingsDatasetLine = SettingsDatasetLine(),
    ):

        super().__init__(data=data, noise_map=noise_map, settings=settings)

        self.data = data
        self.noise_map = noise_map
        self.pre_cti_line = pre_cti_line
        self.layout = layout

    def apply_mask(self, mask: mask_1d.Mask1D) -> "DatasetLine":

        data = Array1D.manual_mask(array=self.data, mask=mask).native
        noise_map = Array1D.manual_mask(array=self.noise_map, mask=mask).native

        return DatasetLine(
            data=data,
            noise_map=noise_map,
            pre_cti_line=self.pre_cti_line,
            layout=self.layout,
        )

    def apply_settings(self, settings: SettingsDatasetLine) -> "DatasetLine":

        return self

    @classmethod
    def from_fits(
        cls,
        layout,
        data_path,
        pixel_scales,
        data_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_single_value=None,
        pre_cti_data_path=None,
        pre_cti_data_hdu=0,
    ):

        data = Array1D.from_fits(
            file_path=data_path, hdu=data_hdu, pixel_scales=pixel_scales
        )

        if noise_map_path is not None:
            noise_map = array_1d_util.numpy_array_1d_from_fits(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            noise_map = np.ones(data.shape_native) * noise_map_from_single_value

        noise_map = Array1D.manual_native(array=noise_map, pixel_scales=pixel_scales)

        if pre_cti_data_path is not None:
            pre_cti_data = Array1D.from_fits(
                file_path=pre_cti_data_path,
                hdu=pre_cti_data_hdu,
                pixel_scales=pixel_scales,
            )
        else:
            if isinstance(layout, Layout1DLine):
                pre_cti_data = layout.pre_cti_data_from(
                    shape_native=data.shape_native, pixel_scales=pixel_scales
                )
            else:
                raise exc.LayoutException(
                    "Cannot estimate pre_cti_data data from non-uniform charge injectiono pattern"
                )

        pre_cti_data = Array1D.manual_native(
            array=pre_cti_data.native, pixel_scales=pixel_scales
        )

        return DatasetLine(
            data=data, noise_map=noise_map, pre_cti_line=pre_cti_data, layout=layout
        )

    def output_to_fits(
        self, data_path, noise_map_path=None, pre_cti_data_path=None, overwrite=False
    ):

        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)
        self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
        self.pre_cti_line.output_to_fits(
            file_path=pre_cti_data_path, overwrite=overwrite
        )


class SimulatorDatasetLine(AbstractSimulatorImaging):
    def __init__(
        self,
        pixel_scales: Union[float, Tuple[float]],
        read_noise: Optional[float] = None,
        add_poisson_noise: bool = False,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the data, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        """

        super().__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.pixel_scales = pixel_scales

    def from_layout(
        self,
        layout: Layout1DLine,
        clocker: Clocker1D,
        traps: Optional[List[AbstractTrap]] = None,
        ccd: Optional[CCDPhase] = None,
    ) -> DatasetLine:
        """Simulate a charge injection data, including effects like noises.

        Parameters
        -----------
        pre_cti_data
        cosmic_ray_map
            The dimensions of the output simulated charge injection data.
        frame_geometry : CIQuadGeometry
            The quadrant geometry of the simulated data, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout : layout_ci.Layout1DLineSimulate
            The charge injection layout_ci (regions, normalization, etc.) of the charge injection data.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise : None or float
            The FWHM of the Gaussian read-noises added to the data.
        noise_seed : int
            Seed for the read-noises added to the data.
        """

        pre_cti_data = layout.pre_cti_data_from(
            shape_native=layout.shape_1d, pixel_scales=self.pixel_scales
        )

        return self.from_pre_cti_data(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=clocker,
            traps=traps,
            ccd=ccd,
        )

    def from_pre_cti_data(
        self,
        pre_cti_data: Array1D,
        layout: Layout1DLine,
        clocker: Clocker1D,
        traps: Optional[List[AbstractTrap]] = None,
        ccd: Optional[CCDPhase] = None,
    ) -> DatasetLine:

        post_cti_data = clocker.add_cti(
            pre_cti_data=pre_cti_data.native, traps=traps, ccd=ccd
        )

        return self.from_post_cti_data(
            post_cti_data=post_cti_data, pre_cti_data=pre_cti_data, layout=layout
        )

    def from_post_cti_data(
        self, post_cti_data: Array1D, pre_cti_data: Array1D, layout: Layout1DLine
    ) -> DatasetLine:

        if self.read_noise is not None:
            data = preprocess.data_with_gaussian_noise_added(
                data=post_cti_data, sigma=self.read_noise, seed=self.noise_seed
            )
            noise_map = (
                self.read_noise
                * Array1D.ones(
                    shape_native=layout.shape_1d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            data = post_cti_data
            noise_map = Array1D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_1d,
                pixel_scales=self.pixel_scales,
            ).native

        return DatasetLine(
            data=Array1D.manual_native(
                array=data.native, pixel_scales=self.pixel_scales
            ),
            noise_map=Array1D.manual_native(
                array=noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_line=Array1D.manual_native(
                array=pre_cti_data.native, pixel_scales=self.pixel_scales
            ),
            layout=layout,
        )
