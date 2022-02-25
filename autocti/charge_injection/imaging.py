import copy
import numpy as np

from arcticpy.src.ccd import CCDPhase
from arcticpy.src.traps import AbstractTrap

import autoarray as aa

from autoarray.dataset.imaging import AbstractSimulatorImaging
from autocti.charge_injection.layout import Layout2DCI
from autocti.mask import mask_2d
from autocti.clocker.two_d import Clocker2D
from autocti import exc

from typing import Union, Optional, List, Tuple


class SettingsImagingCI(aa.SettingsImaging):
    def __init__(
        self,
        parallel_pixels: Tuple[int, int] = None,
        serial_pixels: Tuple[int, int] = None,
    ):

        super().__init__()

        self.parallel_pixels = parallel_pixels
        self.serial_pixels = serial_pixels

    def modify_via_fit_type(self, is_parallel_fit, is_serial_fit):
        """
        Modify the settings based on the type of fit being performed where:

        - If the fit is a parallel only fit (is_parallel_fit=True, is_serial_fit=False) the serial_pixels are set to None
          and all other settings remain the same.

        - If the fit is a serial only fit (is_parallel_fit=False, is_serial_fit=True) the parallel_pixels are set to
          None and all other settings remain the same.

        - If the fit is a parallel and serial fit (is_parallel_fit=True, is_serial_fit=True) the *parallel_pixels* and
          *serial_pixels* are set to None and all other settings remain the same.

         These settings reflect the appropriate way to extract the charge injection imaging data for fits which use a
         parallel only CTI model, serial only CTI model or fit both.

         Parameters
         ----------
         is_parallel_fit : bool
            If True, the CTI model that is used to fit the charge injection data includes a parallel CTI component.
         is_serial_fit : bool
            If True, the CTI model that is used to fit the charge injection data includes a serial CTI component.
        """

        settings = copy.copy(self)

        if is_parallel_fit:
            settings.serial_pixels = None

        if is_serial_fit:
            settings.parallel_pixels = None

        return settings


class ImagingCI(aa.Imaging):
    def __init__(
        self,
        image: aa.Array2D,
        noise_map: aa.Array2D,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        noise_scaling_map_list: Optional[List[aa.Array2D]] = None,
        name=None,
    ):

        super().__init__(image=image, noise_map=noise_map, name=name)

        self.data = self.image.native
        self.noise_map = self.noise_map.native
        self.pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            cosmic_ray_map = cosmic_ray_map.native

        self.cosmic_ray_map = cosmic_ray_map

        if noise_scaling_map_list is not None:
            noise_scaling_map_list = [
                noise_scaling_map.native for noise_scaling_map in noise_scaling_map_list
            ]

        self.noise_scaling_map_list = noise_scaling_map_list

        self.layout = layout

        self.imaging_full = self

    def apply_mask(self, mask: mask_2d.Mask2D) -> "ImagingCI":

        image = aa.Array2D.manual_mask(array=self.image.native, mask=mask)

        noise_map = aa.Array2D.manual_mask(array=self.noise_map.native, mask=mask)

        if self.cosmic_ray_map is not None:

            cosmic_ray_map = aa.Array2D.manual_mask(
                array=self.cosmic_ray_map.native, mask=mask
            )

        else:

            cosmic_ray_map = None

        if self.noise_scaling_map_list is not None:

            noise_scaling_map_list = [
                aa.Array2D.manual_mask(array=noise_scaling_map.native, mask=mask)
                for noise_scaling_map in self.noise_scaling_map_list
            ]
        else:
            noise_scaling_map_list = None

        return ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_data=self.pre_cti_data.native,
            layout=self.layout,
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_list=noise_scaling_map_list,
        )

    def apply_settings(self, settings: SettingsImagingCI):

        if settings.parallel_pixels is not None:

            imaging = self.layout.extract.parallel_calibration.imaging_ci_from(
                imaging_ci=self, columns=settings.parallel_pixels
            )

            mask = self.layout.extract.parallel_calibration.mask_2d_from(
                mask=self.mask, columns=settings.parallel_pixels
            )

        elif settings.serial_pixels is not None:

            imaging = self.layout.extract.serial_calibration.imaging_ci_from(
                imaging_ci=self, rows=settings.serial_pixels
            )

            mask = self.layout.extract.serial_calibration.mask_2d_from(
                mask=self.mask, rows=settings.serial_pixels
            )

        else:

            return self

        imaging = imaging.apply_mask(mask=mask)

        imaging.imaging_full = self.imaging_full

        return imaging

    @property
    def mask(self):
        return self.image.mask

    @classmethod
    def from_fits(
        cls,
        layout,
        image_path,
        pixel_scales,
        image_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_single_value=None,
        pre_cti_data_path=None,
        pre_cti_data_hdu=0,
        pre_cti_data=None,
        cosmic_ray_map_path=None,
        cosmic_ray_map_hdu=0,
    ):

        ci_image = aa.Array2D.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
        )

        if noise_map_path is not None:
            ci_noise_map = aa.util.array_2d.numpy_array_2d_via_fits_from(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_native) * noise_map_from_single_value

        ci_noise_map = aa.Array2D.manual(array=ci_noise_map, pixel_scales=pixel_scales)

        if pre_cti_data_path is not None and pre_cti_data is None:
            pre_cti_data = aa.Array2D.from_fits(
                file_path=pre_cti_data_path,
                hdu=pre_cti_data_hdu,
                pixel_scales=pixel_scales,
            )
        else:
            raise exc.ImagingCIException(
                "Cannot load pre_cti_data from .fits and pass explicit pre_cti_data."
            )

        pre_cti_data = aa.Array2D.manual(
            array=pre_cti_data.native, pixel_scales=pixel_scales
        )

        if cosmic_ray_map_path is not None:

            cosmic_ray_map = aa.Array2D.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                pixel_scales=pixel_scales,
            )

        else:
            cosmic_ray_map = None

        return ImagingCI(
            image=ci_image,
            noise_map=ci_noise_map,
            pre_cti_data=pre_cti_data,
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
        )

    def output_to_fits(
        self,
        image_path,
        noise_map_path=None,
        pre_cti_data_path=None,
        cosmic_ray_map_path=None,
        overwrite=False,
    ):

        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)
        self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
        self.pre_cti_data.output_to_fits(
            file_path=pre_cti_data_path, overwrite=overwrite
        )

        if self.cosmic_ray_map is not None and cosmic_ray_map_path is not None:

            self.cosmic_ray_map.output_to_fits(
                file_path=cosmic_ray_map_path, overwrite=overwrite
            )


class SimulatorImagingCI(AbstractSimulatorImaging):
    def __init__(
        self,
        pixel_scales: aa.type.PixelScales,
        normalization: float,
        max_normalization: float = np.inf,
        column_sigma: Optional[float] = None,
        row_slope: Optional[float] = 0.0,
        read_noise: Optional[float] = None,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
        ci_seed: int = -1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map
            The exposure time of an observation using this data_type.
        """

        super().__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.pixel_scales = pixel_scales

        self.normalization = normalization
        self.max_normalization = max_normalization
        self.column_sigma = column_sigma
        self.row_slope = row_slope

        self.ci_seed = ci_seed

    @property
    def _ci_seed(self):

        if self.ci_seed == -1:
            return np.random.randint(0, int(1e9))
        return self.ci_seed

    def region_ci_from(self, region_dimensions):
        """Generate the non-uniform charge distribution of a charge injection region. This includes non-uniformity \
        across both the rows and columns of the charge injection region.

        Before adding non-uniformity to the rows and columns, we assume an input charge injection level \
        (e.g. the average current being injected). We then simulator non-uniformity in this region.

        Non-uniformity in the columns is caused by sharp peaks and troughs in the input charge current. To simulator  \
        this, we change the normalization of each column by drawing its normalization value from a Gaussian \
        distribution which has a mean of the input normalization and standard deviation *column_sigma*. The seed \
        of the random number generator ensures that the non-uniform charge injection update_via_regions of each pre_cti_datas \
        are identical.

        Non-uniformity in the rows is caused by the charge smoothly decreasing as the injection is switched off. To \
        simulator this, we assume the charge level as a function of row number is not flat but defined by a \
        power-law with slope *row_slope*.

        Non-uniform charge injection images are generated using the function *simulate_pre_cti*, which uses this \
        function.

        Parameters
        -----------
        max_normalization
        column_sigma
        region_dimensions
            The size of the non-uniform charge injection region.
        ci_seed : int
            Input seed for the random number generator to give reproducible results.
        """

        np.random.seed(self._ci_seed)

        ci_rows = region_dimensions[0]
        ci_columns = region_dimensions[1]
        ci_region = np.zeros(region_dimensions)

        for column_number in range(ci_columns):

            column_normalization = 0
            while (
                column_normalization <= 0
                or column_normalization >= self.max_normalization
            ):
                column_normalization = np.random.normal(
                    self.normalization, self.column_sigma
                )

            ci_region[0:ci_rows, column_number] = self.generate_column(
                size=ci_rows,
                normalization=column_normalization,
                row_slope=self.row_slope,
            )

        return ci_region

    def pre_cti_data_uniform_from(self, layout: Layout2DCI):
        """
        Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        """

        pre_cti_data = np.zeros(layout.shape_2d)

        for region in layout.region_list:
            pre_cti_data[region.slice] += self.normalization

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=self.pixel_scales)

    def pre_cti_data_non_uniform_from(self, layout: Layout2DCI) -> aa.Array2D:
        """
        Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by going \
        to its charge injection regions and adding its non-uniform charge distribution.

        For one column of a non-uniform charge injection pre_cti_datas, it is assumed that each non-uniform charge \
        injection region has the same overall normalization value (after drawing this value randomly from a Gaussian \
        distribution). Physically, this is true provided the spikes / troughs in the current that cause \
        non-uniformity occur in an identical fashion for the generation of each charge injection region.

        A non-uniform charge injection layout_ci, which is defined by the regions it appears on a charge injection
        array and its average normalization.

        Non-uniformity across the columns of a charge injection layout_ci is due to spikes / drops in the current that
        injects the charge. This is a noisy process, leading to non-uniformity with no regularity / smoothness. Thus,
        it cannot be modeled with an analytic profile, and must be assumed as prior-knowledge about the charge
        injection electronics or estimated from the observed charge injection ci_data.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope row_slope.

        Parameters
        -----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        ci_seed : int
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        pre_cti_data = np.zeros(layout.shape_2d)

        for region in layout.region_list:
            pre_cti_data[region.slice] += self.region_ci_from(
                region_dimensions=region.shape
            )

        return aa.Array2D.manual(array=pre_cti_data, pixel_scales=self.pixel_scales)

    def generate_column(
        self, size: int, normalization: float, row_slope: float
    ) -> np.ndarray:
        """
        Generate a column of non-uniform charge, including row non-uniformity.

        The pixel-numbering used to generate non-uniformity across the charge injection rows runs from 1 -> size

        Parameters
        -----------
        size : int
            The size of the non-uniform column of charge
        normalization
            The input normalization of the column's charge e.g. the level of charge injected.

        """
        return normalization * (np.arange(1, size + 1)) ** row_slope

    def from_layout(
        self,
        layout: Layout2DCI,
        clocker: Clocker2D,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_trap_list: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        pre_cti_data
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry : CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout : layout_ci.Layout2DCISimulate
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

        if self.column_sigma is not None:
            pre_cti_data = self.pre_cti_data_non_uniform_from(layout=layout)
        else:
            pre_cti_data = self.pre_cti_data_uniform_from(layout=layout)

        return self.from_pre_cti_data(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=clocker,
            parallel_trap_list=parallel_trap_list,
            parallel_ccd=parallel_ccd,
            serial_trap_list=serial_trap_list,
            serial_ccd=serial_ccd,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_pre_cti_data(
        self,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        clocker: Clocker2D,
        parallel_trap_list: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_trap_list: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

        pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            pre_cti_data += cosmic_ray_map.native

        post_cti_data = clocker.add_cti(
            data=pre_cti_data,
            parallel_trap_list=parallel_trap_list,
            parallel_ccd=parallel_ccd,
            serial_trap_list=serial_trap_list,
            serial_ccd=serial_ccd,
        )

        if cosmic_ray_map is not None:
            pre_cti_data -= cosmic_ray_map.native

        return self.from_post_cti_data(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_post_cti_data(
        self,
        post_cti_data: aa.Array2D,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        cosmic_ray_map: Optional[aa.Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

        if self.read_noise is not None:
            ci_image = aa.preprocess.data_with_gaussian_noise_added(
                data=post_cti_data, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = (
                self.read_noise
                * aa.Array2D.ones(
                    shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            ci_image = post_cti_data
            ci_noise_map = aa.Array2D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_2d,
                pixel_scales=self.pixel_scales,
            ).native

        return ImagingCI(
            image=aa.Array2D.manual(
                array=ci_image.native, pixel_scales=self.pixel_scales
            ),
            noise_map=aa.Array2D.manual(
                array=ci_noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_data=aa.Array2D.manual(
                array=pre_cti_data.native, pixel_scales=self.pixel_scales
            ),
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            name=name,
        )
