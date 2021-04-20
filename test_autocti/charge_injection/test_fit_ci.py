import numpy as np
import pytest
import autocti as ac
from autocti.charge_injection.fit_ci import (
    hyper_noise_map_from_noise_map_and_noise_scalings,
)


class TestFitImagingCI:
    def test__fit_quantities_same_as_calculated_individually(
        self, imaging_ci_7x7, layout_ci_7x7, mask_2d_7x7_unmasked
    ):

        masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

        post_cti_ci = ac.ci.CIFrame.ones(
            shape_native=masked_imaging_ci_7x7.image.shape_native,
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging_ci_7x7, post_cti_ci=post_cti_ci
        )

        residual_map = ac.util.fit.residual_map_with_mask_from(
            data=masked_imaging_ci_7x7.image,
            mask=mask_2d_7x7_unmasked,
            model_data=post_cti_ci,
        )

        assert (fit.residual_map == residual_map).all()

        chi_squared_map = ac.util.fit.chi_squared_map_with_mask_from(
            residual_map=residual_map,
            noise_map=masked_imaging_ci_7x7.noise_map,
            mask=mask_2d_7x7_unmasked,
        )

        assert (fit.chi_squared_map == chi_squared_map).all()

        chi_squared = ac.util.fit.chi_squared_with_mask_from(
            chi_squared_map=chi_squared_map, mask=mask_2d_7x7_unmasked
        )

        noise_normalization = ac.util.fit.noise_normalization_with_mask_from(
            noise_map=masked_imaging_ci_7x7.noise_map, mask=mask_2d_7x7_unmasked
        )

        log_likelihood = ac.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert fit.log_likelihood == log_likelihood

    def test__image_and_post_cti_ci_the_same__noise_scaling_all_0s__likelihood_is_noise_normalization(
        self, layout_ci_7x7
    ):

        image = 10.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
        )
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        pre_cti_ci = 10.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
        )

        noise_scaling_maps = [
            ac.ci.CIFrame.zeros(
                shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
            ),
            ac.ci.CIFrame.zeros(
                shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
            ),
        ]

        imaging = ac.ci.ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=pre_cti_ci,
            noise_scaling_maps=noise_scaling_maps,
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        hyper_noise_scalar = ac.ci.HyperCINoiseScalar(scale_factor=1.0)

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging,
            post_cti_ci=pre_cti_ci,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__image_and_post_cti_different__noise_scaling_all_0s__likelihood_chi_squared_and_noise_normalization(
        self, layout_ci_7x7
    ):
        image = ac.ci.CIFrame.full(
            fill_value=9.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=10.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )

        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[0.0, 0.0], [0.0, 0.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            )
        ]

        imaging = ac.ci.ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=pre_cti_ci,
            noise_scaling_maps=noise_scaling_maps,
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        hyper_noise_scalar = ac.ci.HyperCINoiseScalar(scale_factor=1.0)

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging,
            post_cti_ci=pre_cti_ci,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 4.0 * ((1.0 / 2.0) ** 2.0)
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.log_likelihood == pytest.approx(
            -0.5 * (chi_squared + noise_normalization), 1e-4
        )

    def test__image_and_post_cti_ci_the_same__noise_scaling_non_0s__likelihood_is_noise_normalization(
        self, layout_ci_7x7
    ):
        image = 10.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
        )
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=10.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )

        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
            ac.ci.CIFrame.manual(
                array=[[5.0, 6.0], [7.0, 8.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
        ]

        imaging = ac.ci.ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=pre_cti_ci,
            noise_scaling_maps=noise_scaling_maps,
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        hyper_noise_scalar = ac.ci.HyperCINoiseScalar(scale_factor=1.0)

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging,
            post_cti_ci=pre_cti_ci,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (2.0 + 1.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 3.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)
        )

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self, layout_ci_7x7
    ):
        image = 10.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
        )
        noise_map = 3.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=layout_ci_7x7, pixel_scales=1.0
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=10.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )

        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
            ac.ci.CIFrame.manual(
                array=[[5.0, 6.0], [7.0, 8.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
        ]

        imaging = ac.ci.ImagingCI(
            image=image,
            noise_map=noise_map,
            pre_cti_ci=pre_cti_ci,
            noise_scaling_maps=noise_scaling_maps,
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        hyper_noise_scalar_0 = ac.ci.HyperCINoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ac.ci.HyperCINoiseScalar(scale_factor=2.0)

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging,
            post_cti_ci=pre_cti_ci,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (3.0 + 1.0 + 10.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 2.0 + 12.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 3.0 + 14.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 4.0 + 16.0) ** 2.0)
        )

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__all_quantities_are_same_as_calculated_individually(
        self, imaging_ci_7x7, layout_ci_7x7, mask_2d_7x7_unmasked
    ):

        masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

        hyper_noise_scalar_0 = ac.ci.HyperCINoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ac.ci.HyperCINoiseScalar(scale_factor=2.0)

        fit = ac.ci.FitImagingCI(
            imaging_ci=masked_imaging_ci_7x7,
            post_cti_ci=masked_imaging_ci_7x7.pre_cti_ci,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        hyper_noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_map=masked_imaging_ci_7x7.noise_map,
            noise_scaling_maps=masked_imaging_ci_7x7.noise_scaling_maps,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        assert (hyper_noise_map == fit.noise_map).all()

        residual_map = ac.util.fit.residual_map_with_mask_from(
            data=masked_imaging_ci_7x7.image,
            mask=mask_2d_7x7_unmasked,
            model_data=masked_imaging_ci_7x7.pre_cti_ci,
        )

        assert (residual_map == fit.residual_map).all()

        chi_squared_map = ac.util.fit.chi_squared_map_with_mask_from(
            residual_map=residual_map,
            noise_map=hyper_noise_map,
            mask=mask_2d_7x7_unmasked,
        )

        assert (chi_squared_map == fit.chi_squared_map).all()

        chi_squared = ac.util.fit.chi_squared_with_mask_from(
            chi_squared_map=chi_squared_map, mask=mask_2d_7x7_unmasked
        )

        noise_normalization = ac.util.fit.noise_normalization_with_mask_from(
            noise_map=hyper_noise_map, mask=mask_2d_7x7_unmasked
        )

        log_likelihood = ac.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == fit.log_likelihood


class TestHyperNoiseMap:
    def test__image_and_pre_cti_not_identical__noise_scaling_maps_ares__no_noise_map_scaling(
        self, layout_ci_7x7
    ):
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[0.0, 0.0], [0.0, 0.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            )
        ]
        hyper_noise_scalars = [ac.ci.HyperCINoiseScalar(scale_factor=1.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__factor_is__no_noise_map_scaling(
        self, layout_ci_7x7
    ):
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            )
        ]
        hyper_noise_scalars = [ac.ci.HyperCINoiseScalar(scale_factor=0.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_by_noise_map(
        self, layout_ci_7x7
    ):
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            )
        ]
        hyper_noise_scalars = [ac.ci.HyperCINoiseScalar(scale_factor=1.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[3.0, 4.0], [5.0, 6.0]]))).all()

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self, layout_ci_7x7
    ):
        noise_map = ac.ci.CIFrame.full(
            fill_value=2.0,
            shape_native=(2, 2),
            layout_ci=layout_ci_7x7,
            pixel_scales=1.0,
        )
        noise_scaling_maps = [
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
            ac.ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]],
                layout_ci=layout_ci_7x7,
                pixel_scales=1.0,
            ),
        ]
        hyper_noise_scalars = [
            ac.ci.HyperCINoiseScalar(scale_factor=1.0),
            ac.ci.HyperCINoiseScalar(scale_factor=2.0),
        ]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[5.0, 8.0], [11.0, 14.0]]))).all()


class TestChiSquaredMapsOfRegions:
    def test__chi_squared_map_of_regions_ci__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ac.ci.Layout2DCIUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=pattern, pixel_scales=1.0
        )
        noise_map = ac.ci.CIFrame.ones(
            shape_native=(2, 2), layout_ci=pattern, pixel_scales=1.0
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=1.0, shape_native=(2, 2), layout_ci=pattern, pixel_scales=1.0
        )

        imaging = ac.ci.ImagingCI(
            image=image, noise_map=noise_map, pre_cti_ci=pre_cti_ci
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        fit = ac.ci.FitImagingCI(imaging_ci=masked_imaging, post_cti_ci=pre_cti_ci)

        assert (
            fit.chi_squared_map_of_regions_ci == np.array([[4.0, 0.0], [0.0, 0.0]])
        ).all()

    def test__chi_squared_map_of_parallel_non_regions_ci__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ac.ci.Layout2DCIUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_prescan=(1, 2, 1, 2), serial_overscan=(0, 1, 1, 2)),
            pixel_scales=1.0,
        )
        noise_map = ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_prescan=(1, 2, 1, 2), serial_overscan=(0, 1, 1, 2)),
            pixel_scales=1.0,
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=1.0,
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_prescan=(1, 2, 1, 2), serial_overscan=(0, 1, 1, 2)),
            pixel_scales=1.0,
        )

        imaging = ac.ci.ImagingCI(
            image=image, noise_map=noise_map, pre_cti_ci=pre_cti_ci
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        fit = ac.ci.FitImagingCI(imaging_ci=masked_imaging, post_cti_ci=pre_cti_ci)

        assert (
            fit.chi_squared_map_of_parallel_trails == np.array([[0.0, 0.0], [4.0, 0.0]])
        ).all()

    def test__chi_squared_map_of_serial_trails__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ac.ci.Layout2DCIUniform(regions=[(0, 2, 0, 1)], normalization=1.0)

        image = 3.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(1, 2, 0, 2)),
            pixel_scales=1.0,
        )
        noise_map = ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(1, 2, 0, 2)),
            pixel_scales=1.0,
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=1.0,
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(1, 2, 0, 2)),
            pixel_scales=1.0,
        )

        imaging = ac.ci.ImagingCI(
            image=image, noise_map=noise_map, pre_cti_ci=pre_cti_ci
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        fit = ac.ci.FitImagingCI(imaging_ci=masked_imaging, post_cti_ci=pre_cti_ci)

        assert (
            fit.chi_squared_map_of_serial_trails == np.array([[0.0, 4.0], [0.0, 4.0]])
        ).all()

    def test__chi_squared_map_of_overscan_above_serial_trails__extracts_correctly_from_chi_squard_map(
        self,
    ):
        pattern = ac.ci.Layout2DCIUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(0, 2, 1, 2)),
            pixel_scales=1.0,
        )
        noise_map = ac.ci.CIFrame.ones(
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(0, 2, 1, 2)),
            pixel_scales=1.0,
        )
        pre_cti_ci = ac.ci.CIFrame.full(
            fill_value=1.0,
            shape_native=(2, 2),
            layout_ci=pattern,
            scans=ac.Scans(serial_overscan=(0, 2, 1, 2)),
            pixel_scales=1.0,
        )

        imaging = ac.ci.ImagingCI(
            image=image, noise_map=noise_map, pre_cti_ci=pre_cti_ci
        )

        mask = ac.Mask2D.unmasked(shape_native=(2, 2), pixel_scales=1.0)

        masked_imaging = imaging.apply_mask(mask=mask)

        fit = ac.ci.FitImagingCI(imaging_ci=masked_imaging, post_cti_ci=pre_cti_ci)

        assert (
            fit.chi_squared_map_of_serial_overscan_no_trails
            == np.array([[0.0, 0.0], [0.0, 4.0]])
        ).all()
