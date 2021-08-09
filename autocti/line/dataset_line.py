from autoarray.mask import mask_1d
from autoarray.structures.arrays.one_d import array_1d
from autoarray.dataset import abstract_dataset

from arcticpy.src.ccd import CCDPhase
from arcticpy.src.traps import AbstractTrap
from autoarray.structures.arrays.one_d.array_1d import Array1D
from autoarray.dataset import preprocess
from autoarray.dataset.imaging import AbstractSimulatorImaging
from autocti.line.layout_line import Layout1DLine
from autocti.util.clocker import Clocker1D

from typing import Union, Optional, List, Tuple


class SettingsDatasetLine(abstract_dataset.AbstractSettingsDataset):

    pass


class DatasetLine(abstract_dataset.AbstractDataset):
    def __init__(
        self,
        data: array_1d.Array1D,
        noise_map: array_1d.Array1D,
        pre_cti_line: array_1d.Array1D,
        layout: Layout1DLine,
        settings: SettingsDatasetLine = SettingsDatasetLine(),
    ):

        super().__init__(data=data, noise_map=noise_map, settings=settings)

        self.data = data
        self.noise_map = noise_map
        self.pre_cti_line = pre_cti_line
        self.layout = layout

    def apply_mask(self, mask: mask_1d.Mask1D) -> "DatasetLine":

        data = array_1d.Array1D.manual_mask(array=self.data, mask=mask).native
        noise_map = array_1d.Array1D.manual_mask(array=self.noise_map, mask=mask).native

        return DatasetLine(
            data=data,
            noise_map=noise_map,
            pre_cti_line=self.pre_cti_line,
            layout=self.layout,
        )

    def apply_settings(self, settings: SettingsDatasetLine) -> "DatasetLine":

        return self


class SimulatorDatasetLine(AbstractSimulatorImaging):
    def __init__(
        self,
        pixel_scales: Union[float, Tuple[float]],
        read_noise: Optional[float] = None,
        add_poisson_noise: bool = False,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
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
        parallel_traps: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_traps: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[Array1D] = None,
        name: Optional[str] = None,
    ) -> DatasetLine:
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        pre_cti_image
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry : CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout : layout_ci.Layout1DLineSimulate
            The charge injection layout_ci (regions, normalization, etc.) of the charge injection image.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise : None or float
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed : int
            Seed for the read-noises added to the image.
        """

        if isinstance(layout, Layout1DLine):
            pre_cti_image = layout.pre_cti_image_from(
                shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
            )
        else:
            pre_cti_image = layout.pre_cti_image_from(
                shape_native=layout.shape_2d,
                ci_seed=self.ci_seed,
                pixel_scales=self.pixel_scales,
            )

        return self.from_pre_cti_image(
            pre_cti_image=pre_cti_image.native,
            layout=layout,
            clocker=clocker,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_pre_cti_image(
        self,
        pre_cti_image: Array1D,
        layout: Layout1DLine,
        clocker: Clocker1D,
        parallel_traps: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_traps: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[Array1D] = None,
        name: Optional[str] = None,
    ) -> DatasetLine:

        if cosmic_ray_map is not None:
            pre_cti_image += cosmic_ray_map

        post_cti_image = clocker.add_cti(
            image_pre_cti=pre_cti_image.native,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
        )

        if cosmic_ray_map is not None:
            pre_cti_image -= cosmic_ray_map

        return self.from_post_cti_image(
            post_cti_image=post_cti_image,
            pre_cti_image=pre_cti_image,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_post_cti_image(
        self,
        post_cti_image: Array1D,
        pre_cti_image: Array1D,
        layout: Layout1DLine,
        cosmic_ray_map: Optional[Array1D] = None,
        name: Optional[str] = None,
    ) -> DatasetLine:

        if self.read_noise is not None:
            ci_image = preprocess.data_with_gaussian_noise_added(
                data=post_cti_image, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = (
                self.read_noise
                * Array1D.ones(
                    shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            ci_image = post_cti_image
            ci_noise_map = Array1D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_2d,
                pixel_scales=self.pixel_scales,
            ).native

        if cosmic_ray_map is not None:

            cosmic_ray_map = Array1D.manual(
                array=cosmic_ray_map, pixel_scales=self.pixel_scales
            )

        return DatasetLine(
            image=Array1D.manual(array=ci_image.native, pixel_scales=self.pixel_scales),
            noise_map=Array1D.manual(
                array=ci_noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_image=Array1D.manual(
                array=pre_cti_image.native, pixel_scales=self.pixel_scales
            ),
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            name=name,
        )
