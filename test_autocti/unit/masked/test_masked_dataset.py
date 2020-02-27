import numpy as np
import pytest

import autocti as ac


class TestCIMaskedImaging:
    def test__construtor__masks_arrays_correctly(self, ci_imaging_7x7):

        mask = ac.mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)

        mask[0, 0] = True

        masked_ci_imaging = ac.masked.ci_imaging(ci_imaging=ci_imaging_7x7, mask=mask)

        assert (masked_ci_imaging.mask == mask).all()

        masked_image = ci_imaging_7x7.image
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.image == masked_image).all()

        masked_image = ci_imaging_7x7.noise_map
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.noise_map == masked_image).all()

        masked_image = ci_imaging_7x7.ci_pre_cti
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.ci_pre_cti == masked_image).all()

        masked_image = ci_imaging_7x7.cosmic_ray_map
        masked_image[0, 0] = 0.0

        assert (masked_ci_imaging.cosmic_ray_map == masked_image).all()

    def test__for_parallel_masked_ci_imaging(
        self, ci_imaging_7x7, mask_7x7, noise_scaling_maps_list_7x7
    ):

        mask = ac.mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)
        mask[0, 2] = True

        masked_ci_imaging = ac.masked.ci_imaging.for_parallel_from_columns(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            columns=(1, 3),
            noise_scaling_maps_list=noise_scaling_maps_list_7x7,
        )

        mask = np.full(fill_value=False, shape=(7, 2))
        mask[0, 0] = True
        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((7, 2))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((7, 2))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        ci_pre_cti = 10.0 * np.ones((7, 2))
        ci_pre_cti[0, 0] = 0.0

        assert masked_ci_imaging.ci_pre_cti == pytest.approx(ci_pre_cti, 1.0e-4)

        assert masked_ci_imaging.cosmic_ray_map.shape == (7, 2)

        noise_scaling_map_0 = np.ones((7, 2))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps_list[0][0] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )

        noise_scaling_map_0 = 2.0 * np.ones((7, 2))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps_list[0][1] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )

    def test__for_serial_masked_ci_imaging(
        self, ci_imaging_7x7, mask_7x7, noise_scaling_maps_list_7x7
    ):

        mask = ac.mask.unmasked(shape_2d=ci_imaging_7x7.shape_2d)
        mask[1, 0] = True

        masked_ci_imaging = ac.masked.ci_imaging.for_serial_from_rows(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            rows=(0, 1),
            noise_scaling_maps_list=noise_scaling_maps_list_7x7,
        )

        mask = np.full(fill_value=False, shape=(1, 7))
        mask[0, 0] = True
        assert (masked_ci_imaging.mask == mask).all()

        image = np.ones((1, 7))
        image[0, 0] = 0.0

        assert masked_ci_imaging.image == pytest.approx(image, 1.0e-4)

        noise_map = 2.0 * np.ones((1, 7))
        noise_map[0, 0] = 0.0

        assert masked_ci_imaging.noise_map == pytest.approx(noise_map, 1.0e-4)

        ci_pre_cti = 10.0 * np.ones((1, 7))
        ci_pre_cti[0, 0] = 0.0

        assert masked_ci_imaging.ci_pre_cti == pytest.approx(ci_pre_cti, 1.0e-4)

        assert masked_ci_imaging.cosmic_ray_map.shape == (1, 7)

        noise_scaling_map_0 = np.ones((1, 7))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps_list[0][0] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )

        noise_scaling_map_0 = 2.0 * np.ones((1, 7))
        noise_scaling_map_0[0, 0] = 0.0

        assert masked_ci_imaging.noise_scaling_maps_list[0][1] == pytest.approx(
            noise_scaling_map_0, 1.0e-4
        )
