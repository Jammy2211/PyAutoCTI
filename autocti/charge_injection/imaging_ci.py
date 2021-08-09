import copy
import numpy as np

from arcticpy.src.ccd import CCDPhase
from arcticpy.src.traps import AbstractTrap
from autoarray.structures.arrays.two_d.array_2d import Array2D
from autoarray.structures.arrays.two_d import array_2d_util
from autoarray.dataset import preprocess
from autoarray.dataset.imaging import SettingsImaging
from autoarray.dataset.imaging import Imaging
from autoarray.dataset.imaging import AbstractSimulatorImaging
from autocti.charge_injection.layout_ci import Layout2DCI
from autocti.charge_injection.layout_ci import Layout2DCINonUniform
from autocti.mask import mask_2d
from autocti.util.clocker import Clocker2D
from autocti import exc

from typing import Union, Optional, List, Tuple


class SettingsImagingCI(SettingsImaging):
    def __init__(self, parallel_columns=None, serial_rows=None):

        super().__init__()

        self.parallel_columns = parallel_columns
        self.serial_rows = serial_rows

    def modify_via_fit_type(self, is_parallel_fit, is_serial_fit):
        """
        Modify the settings based on the type of fit being performed where:

        - If the fit is a parallel only fit (is_parallel_fit=True, is_serial_fit=False) the serial_rows are set to None
          and all other settings remain the same.

        - If the fit is a serial only fit (is_parallel_fit=False, is_serial_fit=True) the parallel_columns are set to
          None and all other settings remain the same.

        - If the fit is a parallel and serial fit (is_parallel_fit=True, is_serial_fit=True) the *parallel_columns* and
          *serial_rows* are set to None and all other settings remain the same.

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
            settings.serial_rows = None

        if is_serial_fit:
            settings.parallel_columns = None

        return settings


class ImagingCI(Imaging):
    def __init__(
        self,
        image: Array2D,
        noise_map: Array2D,
        pre_cti_image: Array2D,
        layout: Union[Layout2DCI, Layout2DCINonUniform],
        cosmic_ray_map: Optional[Array2D] = None,
        noise_scaling_map_list: Optional[List[Array2D]] = None,
        name=None,
    ):

        super().__init__(image=image, noise_map=noise_map, name=name)

        self.data = self.image.native
        self.noise_map = self.noise_map.native
        self.pre_cti_image = pre_cti_image.native

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

        image = Array2D.manual_mask(array=self.image.native, mask=mask)

        noise_map = Array2D.manual_mask(array=self.noise_map.native, mask=mask)

        if self.cosmic_ray_map is not None:

            cosmic_ray_map = Array2D.manual_mask(
                array=self.cosmic_ray_map.native, mask=mask
            )

        else:

            cosmic_ray_map = None

        if self.noise_scaling_map_list is not None:

            noise_scaling_map_list = [
                Array2D.manual_mask(array=noise_scaling_map.native, mask=mask)
                for noise_scaling_map in self.noise_scaling_map_list
            ]
        else:
            noise_scaling_map_list = None

        return ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_image=self.pre_cti_image.native,
            layout=self.layout,
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_list=noise_scaling_map_list,
        )

    def apply_settings(self, settings):

        if settings.parallel_columns is not None:

            imaging = self.parallel_calibration_imaging_from(
                columns=settings.parallel_columns
            )

            mask = self.layout.mask_for_parallel_calibration_from(
                mask=self.mask, columns=settings.parallel_columns
            )

        elif settings.serial_rows is not None:

            imaging = self.serial_calibration_imaging_for_rows(
                rows=settings.serial_rows
            )

            mask = self.layout.mask_for_serial_calibration_from(
                mask=self.mask, rows=settings.serial_rows
            )

        else:

            return self

        imaging = imaging.apply_mask(mask=mask)

        imaging.imaging_full = self.imaging_full

        return imaging

    @property
    def mask(self):
        return self.image.mask

    def parallel_calibration_imaging_from(self, columns):
        """
        Returnss a function to extract a parallel section for given columns
        """

        cosmic_ray_map = (
            self.layout.array_2d_for_parallel_calibration_from(
                array=self.cosmic_ray_map, columns=columns
            )
            if self.cosmic_ray_map is not None
            else None
        )

        if self.noise_scaling_map_list is not None:

            noise_scaling_map_list = [
                self.layout.array_2d_for_parallel_calibration_from(
                    array=noise_scaling_map, columns=columns
                )
                for noise_scaling_map in self.noise_scaling_map_list
            ]

        else:

            noise_scaling_map_list = None

        extraction_region = self.layout.extraction_region_for_parallel_calibration_from(
            columns=columns
        )

        return ImagingCI(
            image=self.layout.array_2d_for_parallel_calibration_from(
                array=self.image, columns=columns
            ),
            noise_map=self.layout.array_2d_for_parallel_calibration_from(
                array=self.noise_map, columns=columns
            ),
            pre_cti_image=self.layout.array_2d_for_parallel_calibration_from(
                array=self.pre_cti_image, columns=columns
            ),
            layout=self.layout.after_extraction(extraction_region=extraction_region),
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_list=noise_scaling_map_list,
        )

    def serial_calibration_imaging_for_rows(self, rows):
        """
        Returnss a function to extract a serial section for given rows
        """

        cosmic_ray_map = (
            self.layout.array_2d_for_serial_calibration_from(
                array=self.cosmic_ray_map, rows=rows
            )
            if self.cosmic_ray_map is not None
            else None
        )

        if self.noise_scaling_map_list is not None:

            noise_scaling_map_list = [
                self.layout.array_2d_for_serial_calibration_from(
                    array=noise_scaling_map, rows=rows
                )
                for noise_scaling_map in self.noise_scaling_map_list
            ]

        else:

            noise_scaling_map_list = None

        image = self.layout.array_2d_for_serial_calibration_from(
            array=self.image, rows=rows
        )

        return ImagingCI(
            image=image,
            noise_map=self.layout.array_2d_for_serial_calibration_from(
                array=self.noise_map, rows=rows
            ),
            pre_cti_image=self.layout.array_2d_for_serial_calibration_from(
                array=self.pre_cti_image, rows=rows
            ),
            layout=self.layout.extracted_layout_for_serial_calibration_from(
                new_shape_2d=image.shape, rows=rows
            ),
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_map_list=noise_scaling_map_list,
        )

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
        pre_cti_image_path=None,
        pre_cti_image_hdu=0,
        cosmic_ray_map_path=None,
        cosmic_ray_map_hdu=0,
    ):

        ci_image = Array2D.from_fits(
            file_path=image_path, hdu=image_hdu, pixel_scales=pixel_scales
        )

        if noise_map_path is not None:
            ci_noise_map = array_2d_util.numpy_array_2d_from_fits(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_native) * noise_map_from_single_value

        ci_noise_map = Array2D.manual(array=ci_noise_map, pixel_scales=pixel_scales)

        if pre_cti_image_path is not None:
            pre_cti_image = Array2D.from_fits(
                file_path=pre_cti_image_path,
                hdu=pre_cti_image_hdu,
                pixel_scales=pixel_scales,
            )
        else:
            if isinstance(layout, Layout2DCI):
                pre_cti_image = layout.pre_cti_image_from(
                    shape_native=ci_image.shape_native, pixel_scales=pixel_scales
                )
            else:
                raise exc.LayoutException(
                    "Cannot estimate pre_cti_image image from non-uniform charge injectiono pattern"
                )

        pre_cti_image = Array2D.manual(
            array=pre_cti_image.native, pixel_scales=pixel_scales
        )

        if cosmic_ray_map_path is not None:

            cosmic_ray_map = Array2D.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                pixel_scales=pixel_scales,
            )

        else:
            cosmic_ray_map = None

        return ImagingCI(
            image=ci_image,
            noise_map=ci_noise_map,
            pre_cti_image=pre_cti_image,
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
        )

    def output_to_fits(
        self,
        image_path,
        noise_map_path=None,
        pre_cti_image_path=None,
        cosmic_ray_map_path=None,
        overwrite=False,
    ):

        self.image.output_to_fits(file_path=image_path, overwrite=overwrite)
        self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
        self.pre_cti_image.output_to_fits(
            file_path=pre_cti_image_path, overwrite=overwrite
        )

        if self.cosmic_ray_map is not None and cosmic_ray_map_path is not None:

            self.cosmic_ray_map.output_to_fits(
                file_path=cosmic_ray_map_path, overwrite=overwrite
            )


class SimulatorImagingCI(AbstractSimulatorImaging):
    def __init__(
        self,
        pixel_scales: Union[float, Tuple[float, float]],
        read_noise: Optional[float] = None,
        add_poisson_noise: bool = False,
        noise_if_add_noise_false: float = 0.1,
        noise_seed: int = -1,
        ci_seed: int = -1,
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
        self.ci_seed = ci_seed

    def from_layout(
        self,
        layout: Union[Layout2DCI, Layout2DCINonUniform],
        clocker: Clocker2D,
        parallel_traps: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_traps: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        pre_cti_image
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

        if isinstance(layout, Layout2DCI):
            pre_cti_image = layout.pre_cti_image_from(
                shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
            )
        elif isinstance(layout, Layout2DCINonUniform):
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
        pre_cti_image: Array2D,
        layout: Union[Layout2DCI, Layout2DCINonUniform],
        clocker: Clocker2D,
        parallel_traps: Optional[List[AbstractTrap]] = None,
        parallel_ccd: Optional[CCDPhase] = None,
        serial_traps: Optional[List[AbstractTrap]] = None,
        serial_ccd: Optional[CCDPhase] = None,
        cosmic_ray_map: Optional[Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

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
        post_cti_image: Array2D,
        pre_cti_image: Array2D,
        layout: Union[Layout2DCI, Layout2DCINonUniform],
        cosmic_ray_map: Optional[Array2D] = None,
        name: Optional[str] = None,
    ) -> ImagingCI:

        if self.read_noise is not None:
            ci_image = preprocess.data_with_gaussian_noise_added(
                data=post_cti_image, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = (
                self.read_noise
                * Array2D.ones(
                    shape_native=layout.shape_2d, pixel_scales=self.pixel_scales
                ).native
            )
        else:
            ci_image = post_cti_image
            ci_noise_map = Array2D.full(
                fill_value=self.noise_if_add_noise_false,
                shape_native=layout.shape_2d,
                pixel_scales=self.pixel_scales,
            ).native

        if cosmic_ray_map is not None:

            cosmic_ray_map = Array2D.manual(
                array=cosmic_ray_map, pixel_scales=self.pixel_scales
            )

        return ImagingCI(
            image=Array2D.manual(array=ci_image.native, pixel_scales=self.pixel_scales),
            noise_map=Array2D.manual(
                array=ci_noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_image=Array2D.manual(
                array=pre_cti_image.native, pixel_scales=self.pixel_scales
            ),
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
            name=name,
        )
