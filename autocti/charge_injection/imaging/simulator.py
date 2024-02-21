import copy
import numpy as np
from typing import List, Tuple

import autoarray as aa

from autoarray.dataset.imaging.simulator import SimulatorImaging

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.layout import Layout2DCI
from autocti.clocker.two_d import Clocker2D
from autocti.extract.settings import SettingsExtract
from autocti.model.model_util import CTI2D

from typing import Optional


class SimulatorImagingCI(SimulatorImaging):
    def __init__(
        self,
        pixel_scales: aa.type.PixelScales,
        norm: float,
        max_norm: float = np.inf,
        column_sigma: Optional[float] = None,
        row_slope: Optional[float] = 0.0,
        non_uniform_norm_limit=None,
        read_noise: Optional[float] = None,
        charge_noise: Optional[float] = None,
        stray_light : Optional[Tuple[float, float]] = None,
        flat_field_mode : bool = False,
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
            exposure_time=1.0,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

        self.pixel_scales = pixel_scales

        self.norm = norm
        self.max_norm = max_norm
        self.column_sigma = column_sigma
        self.row_slope = row_slope
        self.non_uniform_norm_limit = non_uniform_norm_limit
        self.read_noise = read_noise
        self.charge_noise = charge_noise
        self.stray_light = stray_light
        self.flat_field_mode = flat_field_mode

        self.ci_seed = ci_seed

    @property
    def _ci_seed(self) -> int:
        if self.ci_seed == -1:
            return np.random.randint(0, int(1e9))
        return self.ci_seed

    def median_list_from(self, total_columns: int) -> List[float]:
        np.random.seed(self._ci_seed)

        injection_norm_list = []

        for column_number in range(total_columns):
            injection_norm = 0

            while injection_norm <= 0 or injection_norm >= self.max_norm:
                injection_norm = np.random.normal(self.norm, self.column_sigma)

            injection_norm_list.append(injection_norm)

        return injection_norm_list

    def injection_norm_list_with_limit_from(self, total_columns: int) -> List[float]:
        injection_norm_list = self.median_list_from(
            total_columns=self.non_uniform_norm_limit
        )

        injection_norm_limited_list = []

        for i in range(total_columns):
            injection_norm = np.random.choice(injection_norm_list)

            injection_norm_limited_list.append(injection_norm)

        return injection_norm_limited_list

    def pre_cti_data_uniform_from(self, layout: Layout2DCI) -> aa.Array2D:
        """
        Use this charge injection layout_ci to generate a pre-cti charge injection image. This is performed by \
        going to its charge injection regions and adding the charge injection normalization value.

        Parameters
        ----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        """
        return layout.pre_cti_data_uniform_from(
            norm=self.norm, pixel_scales=self.pixel_scales
        )

    def pre_cti_data_non_uniform_from(self, layout: Layout2DCI) -> aa.Array2D:
        """
        Use this charge injection layout to generate a pre-cti charge injection image. This is performed by going
        to its charge injection regions and adding an input normalization value to each column, which are
        passed in as a list.

        For one column of a non-uniform charge injection pre-cti image, this function assumes that each non-uniform
        charge  injection region has the same overall normalization value. This assumes the spikes / troughs in the
        injection current that cause non-uniformity occur in an identical fashion for each charge injection region.

        Non-uniformity across the rows of a charge injection layout_ci is due to a drop-off in voltage in the current.
        Therefore, it appears smooth and be modeled as an analytic function, which this code assumes is a
        power-law with slope `row_slope`.

        Parameters
        ----------
        shape_native
            The image_shape of the pre_cti_datas to be created.
        row_slope
            The power-law slope of non-uniformity in the row charge injection profile.
        ci_seed
            Input ci_seed for the random number generator to give reproducible results. A new ci_seed is always used for each \
            pre_cti_datas, ensuring each non-uniform ci_region has the same column non-uniformity layout_ci.
        """

        for region in layout.region_list:
            if self.non_uniform_norm_limit is None:
                injection_norm_list = self.median_list_from(
                    total_columns=region.total_columns
                )
            else:
                injection_norm_list = self.injection_norm_list_with_limit_from(
                    total_columns=region.total_columns
                )

        return layout.pre_cti_data_non_uniform_from(
            injection_norm_list=injection_norm_list,
            pixel_scales=self.pixel_scales,
            row_slope=self.row_slope,
        )

    def via_layout_from(
        self,
        layout: Layout2DCI,
        clocker: Optional[Clocker2D],
        cti: Optional[CTI2D],
        cosmic_ray_map: Optional[aa.Array2D] = None,
    ) -> ImagingCI:
        """Simulate a charge injection image, including effects like noises.

        Parameters
        ----------
        pre_cti_data
        cosmic_ray_map
            The dimensions of the output simulated charge injection image.
        frame_geometry
            The quadrant geometry of the simulated image, defining where the parallel / serial overscans are and \
            therefore the direction of clocking and rotations before input into the cti algorithm.
        layout
            The charge injection layout_ci (regions, normalization, etc.) of the charge injection image.
        cti_params
            The CTI model parameters (trap density, trap release_timescales etc.).
        clocker
            The settings that control the cti clocking algorithm (e.g. ccd well_depth express option).
        read_noise
            The FWHM of the Gaussian read-noises added to the image.
        noise_seed
            Seed for the read-noises added to the image.
        """

        if self.flat_field_mode:

            return self.via_flat_field_mode(
                layout=layout,
                clocker=clocker,
                cti=cti,
                cosmic_ray_map=cosmic_ray_map,
            )

        if self.column_sigma is not None:
            pre_cti_data = self.pre_cti_data_non_uniform_from(layout=layout)
        else:
            pre_cti_data = self.pre_cti_data_uniform_from(layout=layout)

        return self.via_pre_cti_data_from(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=clocker,
            cti=cti,
            cosmic_ray_map=cosmic_ray_map,
        )

    def via_pre_cti_data_from(
        self,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        clocker: Optional[Clocker2D],
        cti: Optional[CTI2D],
        cosmic_ray_map: Optional[aa.Array2D] = None,
    ) -> ImagingCI:
        pre_cti_data = pre_cti_data.native

        if cosmic_ray_map is not None:
            pre_cti_data += cosmic_ray_map.native

        if self.stray_light is not None:

            m, c = self.stray_light

            region = layout.extract.parallel_overscan.parallel_overscan

            stray_light = np.array(
                [m * i + c for i in range(0, region.y1)]
            )

            pre_cti_data[0: region.y1, region.x0: region.x1] += stray_light[:, None]


        if self.charge_noise is not None:
            pre_cti_data = layout.extract.parallel_fpr.add_gaussian_noise_to(
                array=pre_cti_data,
                noise_sigma=self.charge_noise,
                noise_seed=self.noise_seed,
                settings=SettingsExtract(
                    pixels_from_end=layout.extract.parallel_fpr.total_rows_min
                ),
            )

        if cti is not None:
            post_cti_data = clocker.add_cti(data=pre_cti_data, cti=cti)
        else:
            post_cti_data = copy.copy(pre_cti_data)

        if cosmic_ray_map is not None:
            pre_cti_data -= cosmic_ray_map.native

        return self.via_post_cti_data_from(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
        )

    def via_post_cti_data_from(
        self,
        post_cti_data: aa.Array2D,
        pre_cti_data: aa.Array2D,
        layout: Layout2DCI,
        cosmic_ray_map: Optional[aa.Array2D] = None,
    ) -> ImagingCI:
        if self.read_noise is not None:
            ci_image = aa.preprocess.data_with_gaussian_noise_added(
                data=post_cti_data, sigma=self.read_noise, seed=self.noise_seed
            )

            ci_image = aa.Array2D.no_mask(
                values=ci_image, pixel_scales=self.pixel_scales
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
            data=aa.Array2D.no_mask(
                values=ci_image.native, pixel_scales=self.pixel_scales
            ),
            noise_map=aa.Array2D.no_mask(
                values=ci_noise_map, pixel_scales=self.pixel_scales
            ),
            pre_cti_data=aa.Array2D.no_mask(
                values=pre_cti_data.native, pixel_scales=self.pixel_scales
            ),
            cosmic_ray_map=cosmic_ray_map,
            layout=layout,
        )

    def via_flat_field_mode(
            self,
            layout: Layout2DCI,
            clocker: Optional[Clocker2D],
            cti: Optional[CTI2D],
            cosmic_ray_map: Optional[aa.Array2D] = None,
    ):

        pre_cti_data = np.zeros(layout.shape_2d)
        pre_cti_data[
            layout.parallel_overscan.x0: layout.parallel_overscan.x1,
            0:layout.parallel_overscan.y0
        ] = self.norm

        pre_cti_data_poisson = np.random.poisson(pre_cti_data, pre_cti_data.shape)

        pre_cti_data_poisson = copy.copy(pre_cti_data_poisson)

        pre_cti_data_poisson = aa.Array2D.no_mask(
            values=pre_cti_data_poisson, pixel_scales=self.pixel_scales
        )

        if cosmic_ray_map is not None:
            pre_cti_data_poisson += cosmic_ray_map.native

        if cti is not None:
            post_cti_data = clocker.add_cti(data=pre_cti_data_poisson, cti=cti)
        else:
            post_cti_data = copy.copy(pre_cti_data)

        pre_cti_data = aa.Array2D.no_mask(
            values=pre_cti_data, pixel_scales=self.pixel_scales
        )

        return self.via_post_cti_data_from(
            post_cti_data=post_cti_data,
            pre_cti_data=pre_cti_data,
            layout=layout,
            cosmic_ray_map=cosmic_ray_map,
        )