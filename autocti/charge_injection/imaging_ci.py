import copy

import numpy as np
from autoconf import conf
from autocti.charge_injection import frame_ci, pattern_ci as pattern
from autoarray.dataset import preprocess, imaging
from autocti.mask import mask_2d
from autoarray.structures.arrays.two_d import array_2d_util
from autocti import exc


class SettingsCIImaging(imaging.SettingsImaging):
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


class CIImaging(imaging.Imaging):
    def __init__(
        self,
        image,
        noise_map,
        pre_cti_ci,
        cosmic_ray_map=None,
        noise_scaling_maps=None,
        name=None,
    ):

        super().__init__(image=image, noise_map=noise_map, name=name)

        self.pre_cti_ci = pre_cti_ci
        self.cosmic_ray_map = cosmic_ray_map
        self.noise_scaling_maps = noise_scaling_maps

        self.imaging_ci_full = self

    def apply_mask(self, mask: mask_2d.Mask2D):

        image = frame_ci.CIFrame.from_frame_ci(frame_ci=self.image, mask=mask)
        noise_map = frame_ci.CIFrame.from_frame_ci(frame_ci=self.noise_map, mask=mask)

        if self.cosmic_ray_map is not None:

            cosmic_ray_map = frame_ci.CIFrame.from_frame_ci(
                frame_ci=self.cosmic_ray_map, mask=mask
            )

        else:

            cosmic_ray_map = None

        if self.noise_scaling_maps is not None:
            noise_scaling_maps = [
                frame_ci.CIFrame.from_frame_ci(frame_ci=noise_scaling_map, mask=mask)
                for noise_scaling_map in self.noise_scaling_maps
            ]
        else:
            noise_scaling_maps = None

        return CIImaging(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=self.pre_cti_ci,
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_maps=noise_scaling_maps,
        )

    def apply_settings(self, settings):

        imaging_ci_full = copy.deepcopy(self)

        if settings.parallel_columns is not None:

            imaging_ci = self.parallel_calibration_imaging_ci_for_columns(
                columns=settings.parallel_columns
            )

            mask = self.image.parallel_calibration_mask_from_mask_and_columns(
                mask=self.mask, columns=settings.parallel_columns
            )

        elif settings.serial_rows is not None:

            imaging_ci = self.serial_calibration_imaging_ci_for_rows(
                rows=settings.serial_rows
            )

            mask = self.image.serial_calibration_mask_from_mask_and_rows(
                mask=self.mask, rows=settings.serial_rows
            )

        else:

            return self

        imaging_ci = imaging_ci.apply_mask(mask=mask)

        imaging_ci.imaging_ci_full = imaging_ci_full

        return imaging_ci

    @property
    def imaging_ci(self):
        return self.imaging

    @property
    def mask(self):
        return self.image.mask

    @property
    def pattern_ci(self):
        return self.image.pattern_ci

    def parallel_calibration_imaging_ci_for_columns(self, columns):
        """
        Returnss a function to extract a parallel section for given columns
        """

        cosmic_ray_map = (
            self.cosmic_ray_map.parallel_calibration_frame_from_columns(columns=columns)
            if self.cosmic_ray_map is not None
            else None
        )

        if self.noise_scaling_maps is not None:

            noise_scaling_maps = [
                noise_scaling_map.parallel_calibration_frame_from_columns(
                    columns=columns
                )
                for noise_scaling_map in self.noise_scaling_maps
            ]

        else:

            noise_scaling_maps = None

        return CIImaging(
            image=self.image.parallel_calibration_frame_from_columns(columns=columns),
            noise_map=self.noise_map.parallel_calibration_frame_from_columns(
                columns=columns
            ),
            pre_cti_ci=self.pre_cti_ci.parallel_calibration_frame_from_columns(
                columns=columns
            ),
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_maps=noise_scaling_maps,
        )

    def serial_calibration_imaging_ci_for_rows(self, rows):
        """
        Returnss a function to extract a serial section for given rows
        """

        cosmic_ray_map = (
            self.cosmic_ray_map.serial_calibration_frame_from_rows(rows=rows)
            if self.cosmic_ray_map is not None
            else None
        )

        if self.noise_scaling_maps is not None:

            noise_scaling_maps = [
                noise_scaling_map.serial_calibration_frame_from_rows(rows=rows)
                for noise_scaling_map in self.noise_scaling_maps
            ]

        else:

            noise_scaling_maps = None

        return CIImaging(
            image=self.image.serial_calibration_frame_from_rows(rows=rows),
            noise_map=self.noise_map.serial_calibration_frame_from_rows(rows=rows),
            pre_cti_ci=self.pre_cti_ci.serial_calibration_frame_from_rows(rows=rows),
            cosmic_ray_map=cosmic_ray_map,
            noise_scaling_maps=noise_scaling_maps,
        )

    @classmethod
    def from_fits(
        cls,
        pattern_ci,
        image_path,
        pixel_scales,
        roe_corner=(1, 0),
        scans=None,
        image_hdu=0,
        noise_map_path=None,
        noise_map_hdu=0,
        noise_map_from_single_value=None,
        pre_cti_ci_path=None,
        pre_cti_ci_hdu=0,
        cosmic_ray_map_path=None,
        cosmic_ray_map_hdu=0,
    ):

        ci_image = frame_ci.CIFrame.from_fits(
            file_path=image_path,
            hdu=image_hdu,
            roe_corner=roe_corner,
            pattern_ci=pattern_ci,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if noise_map_path is not None:
            ci_noise_map = array_2d_util.numpy_array_2d_from_fits(
                file_path=noise_map_path, hdu=noise_map_hdu
            )
        else:
            ci_noise_map = np.ones(ci_image.shape_native) * noise_map_from_single_value

        ci_noise_map = frame_ci.CIFrame.manual(
            array=ci_noise_map,
            roe_corner=roe_corner,
            pattern_ci=pattern_ci,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if pre_cti_ci_path is not None:
            pre_cti_ci = array_2d_util.numpy_array_2d_from_fits(
                file_path=pre_cti_ci_path, hdu=pre_cti_ci_hdu
            )
        else:
            if isinstance(pattern_ci, pattern.CIPatternUniform):
                pre_cti_ci = pattern_ci.pre_cti_ci_from(
                    shape_native=ci_image.shape, pixel_scales=pixel_scales
                )
            else:
                raise exc.CIPatternException(
                    "Cannot estimate pre_cti_ci image from non-uniform charge injectiono pattern"
                )

        pre_cti_ci = frame_ci.CIFrame.manual(
            array=pre_cti_ci,
            roe_corner=roe_corner,
            pattern_ci=pattern_ci,
            pixel_scales=pixel_scales,
            scans=scans,
        )

        if cosmic_ray_map_path is not None:

            cosmic_ray_map = frame_ci.CIFrame.from_fits(
                file_path=cosmic_ray_map_path,
                hdu=cosmic_ray_map_hdu,
                roe_corner=roe_corner,
                pattern_ci=pattern_ci,
                pixel_scales=pixel_scales,
                scans=scans,
            )

        else:
            cosmic_ray_map = None

        return CIImaging(
            image=ci_image,
            noise_map=ci_noise_map,
            pre_cti_ci=pre_cti_ci,
            cosmic_ray_map=cosmic_ray_map,
        )

    def output_to_fits(
        self,
        image_path,
        noise_map_path=None,
        pre_cti_ci_path=None,
        cosmic_ray_map_path=None,
        overwrite=False,
    ):

        array_2d_util.numpy_array_2d_to_fits(
            array_2d=self.image, file_path=image_path, overwrite=overwrite
        )

        if self.noise_map is not None and noise_map_path is not None:
            array_2d_util.numpy_array_2d_to_fits(
                array_2d=self.noise_map, file_path=noise_map_path, overwrite=overwrite
            )

        if self.pre_cti_ci is not None and pre_cti_ci_path is not None:
            array_2d_util.numpy_array_2d_to_fits(
                array_2d=self.pre_cti_ci, file_path=pre_cti_ci_path, overwrite=overwrite
            )

        if self.cosmic_ray_map is not None and cosmic_ray_map_path is not None:
            array_2d_util.numpy_array_2d_to_fits(
                array_2d=self.cosmic_ray_map,
                file_path=cosmic_ray_map_path,
                overwrite=overwrite,
            )


class SimulatorCIImaging(imaging.AbstractSimulatorImaging):
    def __init__(
        self,
        shape_native,
        pixel_scales,
        read_noise=None,
        add_poisson_noise=False,
        scans=None,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
        ci_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        """

        super(SimulatorCIImaging, self).__init__(
            read_noise=read_noise,
            exposure_time=1.0,
            add_poisson_noise=add_poisson_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.shape_native = shape_native
        self.scans = scans
        self.pixel_scales = pixel_scales
        self.ci_seed = ci_seed

    def from_pattern_ci(
        self,
        pattern_ci,
        clocker,
        parallel_traps=None,
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
        cosmic_ray_map=None,
        name=None,
    ):
        """Simulate a charge injection image, including effects like noises.

        Parameters
        -----------
        pre_cti_ci
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry : frame_ci.CIQuadGeometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        pattern_ci : pattern_ci.CIPatternSimulate
            The charge injection pattern_ci (regions, normalization, etc.) of the charge injection image.
        cti_params : ArcticParams.ArcticParams
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker : ArcticSettings.ArcticSettings
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise : None or float
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed : int
            Seed for the read-noises added to the image.
        """
        if isinstance(pattern_ci, pattern.CIPatternUniform):
            pre_cti_ci = pattern_ci.pre_cti_ci_from(
                shape_native=self.shape_native, pixel_scales=self.pixel_scales
            )
        else:
            pre_cti_ci = pattern_ci.pre_cti_ci_from(
                shape_native=self.shape_native,
                ci_seed=self.ci_seed,
                pixel_scales=self.pixel_scales,
            )

        return self.from_pre_cti_ci(
            pre_cti_ci=pre_cti_ci,
            pattern_ci=pattern_ci,
            clocker=clocker,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_pre_cti_ci(
        self,
        pre_cti_ci,
        pattern_ci,
        clocker,
        parallel_traps=None,
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
        cosmic_ray_map=None,
        name=None,
    ):

        if cosmic_ray_map is not None:
            pre_cti_ci += cosmic_ray_map

        post_cti_ci = clocker.add_cti(
            image=pre_cti_ci,
            parallel_traps=parallel_traps,
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
        )

        return self.from_post_cti_ci(
            post_cti_ci=post_cti_ci,
            pre_cti_ci=pre_cti_ci,
            pattern_ci=pattern_ci,
            cosmic_ray_map=cosmic_ray_map,
            name=name,
        )

    def from_post_cti_ci(
        self, post_cti_ci, pre_cti_ci, pattern_ci, cosmic_ray_map=None, name=None
    ):

        if self.read_noise is not None:
            ci_image = preprocess.data_with_gaussian_noise_added(
                data=post_cti_ci, sigma=self.read_noise, seed=self.noise_seed
            )
            ci_noise_map = self.read_noise * np.ones(post_cti_ci.shape)
        else:
            ci_image = post_cti_ci
            ci_noise_map = None

        return CIImaging(
            image=frame_ci.CIFrame.manual(
                array=ci_image,
                pattern_ci=pattern_ci,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            noise_map=frame_ci.CIFrame.manual(
                array=ci_noise_map,
                pattern_ci=pattern_ci,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            pre_cti_ci=frame_ci.CIFrame.manual(
                array=pre_cti_ci,
                pattern_ci=pattern_ci,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            cosmic_ray_map=frame_ci.CIFrame.manual(
                array=cosmic_ray_map,
                pattern_ci=pattern_ci,
                scans=self.scans,
                pixel_scales=self.pixel_scales,
            ),
            name=name,
        )
