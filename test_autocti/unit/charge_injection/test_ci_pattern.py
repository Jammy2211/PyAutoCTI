import numpy as np
import pytest
import autocti as ac
from autocti import exc


class TestCIPattern(object):
    def test__total_rows_minimum(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 2, 0, 1)])

        assert pattern.total_rows_min == 1

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 1)])

        assert pattern.total_rows_min == 2

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 2, 0, 1), (3, 4, 0, 1)]
        )

        assert pattern.total_rows_min == 1

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 2, 0, 1), (3, 5, 0, 1)]
        )

        assert pattern.total_rows_min == 1

    def test__total_columns_minimum(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 2)])

        assert pattern.total_columns_min == 1

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 3)])

        assert pattern.total_columns_min == 2

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 2), (0, 1, 3, 4)]
        )

        assert pattern.total_columns_min == 1

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 2), (0, 1, 3, 5)]
        )

        assert pattern.total_columns_min == 1

    def test__rows_between_regions(self):

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(1, 2, 1, 2)])

        assert pattern.rows_between_regions == []

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 2, 1, 2), (3, 4, 3, 4)]
        )

        assert pattern.rows_between_regions == [1]

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 2, 1, 2), (4, 5, 3, 4)]
        )

        assert pattern.rows_between_regions == [2]

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=[(1, 2, 1, 2), (4, 5, 3, 4), (8, 9, 3, 4)]
        )

        assert pattern.rows_between_regions == [2, 3]

    def test__check_pattern_dimensions__pattern_has_more_rows_than_image__1_region(
        self,
    ):
        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=([(0, 3, 0, 1)]))

        with pytest.raises(exc.CIPatternException):
            pattern.check_pattern_is_within_image_dimensions(dimensions=(2, 6))

        pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=([(0, 1, 0, 3)]))

        with pytest.raises(exc.CIPatternException):
            pattern.check_pattern_is_within_image_dimensions(dimensions=(6, 2))

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=([(0, 3, 0, 1), (0, 1, 0, 3)])
        )

        with pytest.raises(exc.CIPatternException):
            pattern.check_pattern_is_within_image_dimensions(dimensions=(2, 6))

        pattern = ac.ci.CIPatternUniform(
            normalization=1.0, regions=([(0, 3, 0, 1), (0, 1, 0, 3)])
        )

        with pytest.raises(exc.CIPatternException):
            pattern.check_pattern_is_within_image_dimensions(dimensions=(6, 2))

        with pytest.raises(exc.RegionException):
            ac.ci.CIPatternUniform(normalization=1.0, regions=([(-1, 0, 0, 0)]))

        with pytest.raises(exc.RegionException):
            ac.ci.CIPatternUniform(normalization=1.0, regions=([(0, -1, 0, 0)]))

        with pytest.raises(exc.RegionException):
            ac.ci.CIPatternUniform(normalization=1.0, regions=([(0, 0, -1, 0)]))

        with pytest.raises(exc.RegionException):
            ac.ci.CIPatternUniform(normalization=1.0, regions=([(0, 0, 0, -1)]))

        def test__with_extracted_regions__regions_are_extracted_correctly(self):

            pattern = ac.ci.CIPatternUniform(normalization=1.0, regions=[(0, 2, 0, 2)])

            pattern_extracted = pattern.with_extracted_regions(
                extraction_region=ac.Region((0, 2, 0, 2))
            )

            assert pattern_extracted.regions == [(0, 2, 0, 2)]

            pattern_extracted = pattern.with_extracted_regions(
                extraction_region=ac.Region((0, 1, 0, 1))
            )

            assert pattern_extracted.regions == [(0, 1, 0, 1)]

            pattern = ac.ci.CIPatternUniform(
                normalization=1.0, regions=[(2, 4, 2, 4), (0, 1, 0, 1)]
            )

            pattern_extracted = pattern.with_extracted_regions(
                extraction_region=ac.Region((0, 3, 0, 3))
            )

            assert pattern_extracted.regions == [(2, 3, 2, 3), (0, 1, 0, 1)]

            pattern_extracted = pattern.with_extracted_regions(
                extraction_region=ac.Region((2, 5, 2, 5))
            )

            assert pattern_extracted.regions == [(0, 2, 0, 2)]

            pattern_extracted = pattern.with_extracted_regions(
                extraction_region=ac.Region((8, 9, 8, 9))
            )

            assert pattern_extracted.regions == None


class TestCIPatternUniform(object):
    def test__ci_pre_cti_from_shape_2d__image_3x3__1_ci_region(self):
        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 2)])

        ci_pre_cti = pattern.ci_pre_cti_from(shape_2d=(3, 3), pixel_scales=1.0)

        assert (
            ci_pre_cti
            == np.array([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

    def test__ci_pre_cti_from_shape_2d__image_3x3__2_ci_regions(self):
        ci_pattern_uni = ac.ci.CIPatternUniform(
            normalization=20.0, regions=[(0, 2, 0, 2), (2, 3, 2, 3)]
        )
        image1 = ci_pattern_uni.ci_pre_cti_from(shape_2d=(3, 3), pixel_scales=1.0)

        assert (
            image1 == np.array([[20.0, 20.0, 0.0], [20.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        ).all()

        ci_pattern_uni = ac.ci.CIPatternUniform(
            normalization=30.0, regions=[(0, 3, 0, 2), (2, 3, 2, 3)]
        )
        image1 = ci_pattern_uni.ci_pre_cti_from(shape_2d=(4, 3), pixel_scales=1.0)

        assert (
            image1
            == np.array(
                [
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 30.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__ci_pre_cti_from_shape_2d__pattern_bigger_than_image_dimensions__raises_error(
        self,
    ):
        pattern = ac.ci.CIPatternUniform(normalization=10.0, regions=[(0, 2, 0, 1)])

        with pytest.raises(exc.CIPatternException):
            pattern.ci_pre_cti_from(shape_2d=(1, 1), pixel_scales=1.0)


class TestCIPatternNonUniform(object):
    def test__ci_region_from__uniform_column_and_uniform_row__returns_uniform_charge_region(
        self,
    ):
        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=100.0, regions=[(0, 1, 0, 1)], row_slope=0.0, column_sigma=0.0
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        assert (
            region
            == np.array(
                [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
            )
        ).all()

        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=500.0, regions=[(0, 1, 0, 1)], row_slope=0.0, column_sigma=0.0
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        assert (
            region
            == np.array(
                [
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_column_and_uniform_row__returns_region(self):
        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=100.0, regions=[(0, 1, 0, 1)], row_slope=0.0, column_sigma=1.0
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [101.6, 99.4, 99.5], [101.6, 99.4, 99.5]])
        ).all()

        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=500.0, regions=[(0, 1, 0, 1)], row_slope=0.0, column_sigma=1.0
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                ]
            )
        ).all()

    def test__ci_region_from__uniform_column_and_non_uniform_row__returns_region(self):
        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[100.0, 100.0, 100.0], [99.3, 99.3, 99.3], [98.9, 98.9, 98.9]])
        ).all()

        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=500.0,
            regions=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [500.0, 500.0, 500.0],
                    [496.5, 496.5, 496.5],
                    [494.5, 494.5, 494.5],
                    [493.1, 493.1, 493.1],
                    [492.0, 492.0, 492.0],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_column_and_non_uniform_row__returns_region(
        self,
    ):
        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [100.9, 98.7, 98.8], [100.5, 98.3, 98.4]])
        ).all()

        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=500.0,
            regions=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [501.6, 499.4, 499.5],
                    [498.2, 495.9, 496.0],
                    [496.1, 493.9, 494.0],
                    [494.7, 492.5, 492.6],
                    [493.6, 491.4, 491.5],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_columns_with_large_deviation_value__no_negative_charge_columns_are_generated(
        self,
    ):
        ci_pattern = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=100.0,
        )

        region = ci_pattern.ci_region_from_region(region_dimensions=(10, 10), ci_seed=1)

        assert (region > 0).all()

    def test__ci_pre_cti_from__no_non_uniformity__identical_to_uniform_image__one_ci_region(
        self,
    ):
        ci_pattern_uni = ac.ci.CIPatternUniform(
            normalization=10.0, regions=[(2, 4, 0, 5)]
        )
        image1 = ci_pattern_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=10.0, regions=[(2, 4, 0, 5)], row_slope=0.0, column_sigma=0.0
        )
        image2 = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        assert (image1 == image2).all()

        ci_pattern_uni = ac.ci.CIPatternUniform(
            normalization=100.0, regions=[(1, 4, 2, 5)]
        )
        image1 = ci_pattern_uni.ci_pre_cti_from(shape_2d=(5, 7), pixel_scales=1.0)

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0, regions=[(1, 4, 2, 5)], row_slope=0.0, column_sigma=0.0
        )
        image2 = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 7), pixel_scales=1.0)

        assert (image1 == image2).all()

        ci_pattern_uni = ac.ci.CIPatternUniform(
            normalization=100.0, regions=[(0, 2, 0, 2), (2, 3, 0, 5)]
        )
        image1 = ci_pattern_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 2, 0, 2), (2, 3, 0, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        image2 = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        assert (image1 == image2).all()

    def test__ci_pre_cti_from__non_uniformity_in_columns_only__one_ci_region__image_is_correct(
        self,
    ):
        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0, regions=[(0, 3, 0, 3)], row_slope=0.0, column_sigma=1.0
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(
            shape_2d=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [101.6, 99.4, 99.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0, regions=[(1, 4, 1, 4)], row_slope=0.0, column_sigma=1.0
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(
            shape_2d=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(1, 4, 1, 3), (1, 4, 4, 5)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(
            shape_2d=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 0.0, 101.6],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__ci_pre_cti_from__non_uniformity_in_columns_only__maximum_normalization_input__does_not_simulate_above(
        self,
    ):
        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 5, 0, 5)],
            row_slope=0.0,
            column_sigma=100.0,
            maximum_normalization=100.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(
            shape_2d=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        # Checked ci_seed to ensure the max is above 100.0 without a maximum_normalization
        assert np.max(image) < 100.0

    def test__ci_pre_cti_from__non_uniformity_in_rows_only__one_ci_region__image_is_correct(
        self,
    ):
        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 3, 0, 3)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [100.0, 100.0, 100.0, 0.0, 0.0],
                    [99.3, 99.3, 99.3, 0.0, 0.0],
                    [98.9, 98.9, 98.9, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 100.0],
                    [0.0, 100.0, 100.0, 100.0, 99.3],
                    [0.0, 99.3, 99.3, 99.3, 98.9],
                    [0.0, 98.9, 98.9, 98.9, 98.6],
                    [0.0, 98.6, 98.6, 98.6, 98.4],
                ]
            )
        ).all()

    def test__ci_pre_cti_from__non_uniformity_in_rows_and_columns__two_ci_regions__image_is_correct(
        self,
    ):
        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(
            shape_2d=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 101.6],
                    [0.0, 101.6, 99.4, 99.5, 100.9],
                    [0.0, 100.9, 98.7, 98.8, 100.5],
                    [0.0, 100.5, 98.3, 98.4, 100.2],
                    [0.0, 100.2, 98.0, 98.1, 100.0],
                ]
            )
        ).all()

        ci_pattern_non_uni = ac.ci.CIPatternNonUniform(
            normalization=100.0,
            regions=[(0, 2, 0, 5), (3, 5, 0, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = ci_pattern_non_uni.ci_pre_cti_from(shape_2d=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (image[0:2, 0:5] == image[3:5, 0:5]).all()
