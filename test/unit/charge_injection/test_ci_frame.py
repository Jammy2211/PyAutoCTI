import numpy as np

from autocti.charge_injection import ci_frame, ci_pattern
from autocti.data import mask as msk
from test.unit.mock.mock import MockCIGeometry


class TestBinArrayAcrossSerial:
    def test__3x3_array__all_1s__bin_gives_a_1d_array_of_3_1s(self):
        image = np.ones((3, 3))

        binned_array = ci_frame.bin_array_across_serial(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0])).all()

    def test__4x3_array__all_1s__bin_gives_a_1d_array_of_4_1s(self):
        image = np.ones((4, 3))

        binned_array = ci_frame.bin_array_across_serial(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__3x4_array__all_1s__bin_gives_a_1d_array_of_3_1s(self):
        image = np.ones((3, 4))

        binned_array = ci_frame.bin_array_across_serial(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0])).all()

    def test__3x3_array__different_values__bin_gives_a_1d_array_of_each_mean(self):
        image = np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])

        binned_array = ci_frame.bin_array_across_serial(image)

        assert (binned_array == np.array([2.0, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_including_mask(self):
        image = np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])

        mask = np.ma.array(
            [[False, False, True], [False, False, False], [False, False, False]]
        )

        binned_array = ci_frame.bin_array_across_serial(image, mask)

        assert (binned_array == np.array([1.5, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_an_entire_row_is_masked(self):
        image = np.array([[1.0, 2.0, 3.0], [6.0, 6.0, 6.0], [9.0, 9.0, 9.0]])

        mask = np.ma.array(
            [[False, False, True], [True, True, True], [False, False, False]]
        )

        binned_array = ci_frame.bin_array_across_serial(image, mask)

        assert (binned_array == np.array([1.5, np.inf, 9.0])).all()


class TestBinArrayAcrossParallel:
    def test__3x3_array__all_1s__bin_gives_a_1d_array_of_3_1s(self):
        image = np.ones((3, 3))

        binned_array = ci_frame.bin_array_across_parallel(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0])).all()

    def test__4x3_array__all_1s__bin_gives_a_1d_array_of_3_1s(self):
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        image = np.ones((4, 3))

        binned_array = ci_frame.bin_array_across_parallel(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0])).all()

    def test__3x4_array__all_1s__bin_gives_a_1d_array_of_4_1s(self):
        pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        image = np.ones((3, 4))

        binned_array = ci_frame.bin_array_across_parallel(image)

        assert (binned_array == np.array([1.0, 1.0, 1.0, 1.0])).all()

    def test__3x3_array__different_values__bin_gives_a_1d_array_of_each_mean(self):
        image = np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])

        binned_array = ci_frame.bin_array_across_parallel(image)

        assert (binned_array == np.array([2.0, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_with_mask(self):
        image = np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])

        mask = np.ma.array(
            [[False, False, False], [False, False, False], [True, False, False]]
        )

        binned_array = ci_frame.bin_array_across_parallel(image, mask)

        assert (binned_array == np.array([1.5, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_with_entire_column_masked(self):
        image = np.array([[1.0, 6.0, 9.0], [2.0, 6.0, 9.0], [3.0, 6.0, 9.0]])

        mask = np.ma.array(
            [[False, True, False], [False, True, False], [True, True, False]]
        )

        binned_array = ci_frame.bin_array_across_parallel(image, mask)

        assert (binned_array == np.array([1.5, np.inf, 9.0])).all()


class TestChInj(object):
    class TestCiRegionArrayFromFrame:
        def test__1_ci_region__extracted_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])
            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.ci_regions_from_array(image)

            assert (
                new_frame
                == np.array(
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
                )
            ).all()

        def test__2_ci_regions__extracted_correctly(self):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 1, 2), (2, 3, 1, 3)]
            )
            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.ci_regions_from_array(image)

            assert (
                new_frame
                == np.array(
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
                )
            ).all()

    class TestCIParallelNonRegionArrayFromFrame:
        def test__1_ci_region__parallel_overscan_is_entire_image__extracts_everything_between_its_columns(
            self
        ):

            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(
                frame_geometry=MockCIGeometry(parallel_overscan=(3, 4, 0, 3)),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_non_ci_regions_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [9.0, 10.0, 11.0],
                    ]
                )
            ).all()

        def test__same_as_above_but_2_ci_regions(self):

            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=MockCIGeometry(parallel_overscan=(3, 4, 0, 3)),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_non_ci_regions_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0],
                        [0.0, 0.0, 0.0],
                        [12.0, 13.0, 14.0],
                    ]
                )
            ).all()

        def test__same_as_above_with_parallel_overscan_thinner__columsn_outside_overscan_are_zeros(
            self
        ):

            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=MockCIGeometry(parallel_overscan=(3, 4, 1, 2)),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_non_ci_regions_frame_from_frame(image)

            print(new_frame)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 4.0, 0.0],
                        [0.0, 7.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 13.0, 0.0],
                    ]
                )
            ).all()

    class TestParallelEdgesAndTrailsArrayFromFrame:
        def test__front_edge_only__1_row__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, front_edge_rows=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                )
            ).all()

        def test__front_edge_only__2_rows__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, front_edge_rows=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                )
            ).all()

        def test__trails_only__1_row__new_frame_is_just_that_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, trails_rows=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [12.0, 13.0, 14.0],
                    ]
                )
            ).all()

        def test__trails_only__2_rows__new_frame_is_the_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, trails_rows=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0],
                    ]
                )
            ).all()

        def test__front_edge_and_trails__2_rows_of_each__new_frame_is_edge_and_trail(
            self
        ):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, front_edge_rows=(0, 2), trails_rows=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0],
                    ]
                )
            ).all()

        def test__front_edge_and_trails__2_regions__1_row_of_each__new_frame_is_edge_and_trail(
            self
        ):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(
                image, front_edge_rows=(0, 1), trails_rows=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [3.0, 4.0, 5.0],
                        [0.0, 0.0, 0.0],
                        [9.0, 10.0, 11.0],
                        [12.0, 13.0, 14.0],
                        [0.0, 0.0, 0.0],
                    ]
                )
            ).all()

    class TesParallelCalibrationSectionFromFrame:
        def test__geometry_left__columns_0_to_1__extracts_1_column_left_hand_side_of_array(
            self
        ):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            extracted_side = frame.parallel_calibration_section_for_columns(
                image, columns=(0, 1)
            )

            assert (
                extracted_side == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])
            ).all()

        def test__geometry_bottom__columns_1_to_3__extracts_2_columns_middle_and_right_of_array(
            self
        ):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 5, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            extracted_side = frame.parallel_calibration_section_for_columns(
                image, columns=(1, 3)
            )

            assert (
                extracted_side
                == np.array(
                    [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
                )
            ).all()

        def test__geometry_right__columns_1_to_3__extracts_2_columns_middle_and_left_of_array(
            self
        ):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 5, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                    [0.0, 1.0, 2.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            extracted_side = frame.parallel_calibration_section_for_columns(
                image, columns=(1, 3)
            )

            assert (
                extracted_side
                == np.array(
                    [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
                )
            ).all()

    class TestKeepSerialEdgesAndTrailsArrayFromFrame:
        def test__front_edge_only__1_column__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, front_edge_columns=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
                )
            ).all()

        def test__front_edge_only__2_columns__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, front_edge_columns=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]]
                )
            ).all()

        def test__trails_only__1_column__new_frame_is_just_that_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, trails_columns=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]]
                )
            ).all()

        def test__trails_only__2_columns__new_frame_is_the_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, trails_columns=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
                )
            ).all()

        def test__front_edge_and_trails__2_columns_of_each__new_frame_is_edge_and_trail(
            self
        ):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            image = np.array(
                [
                    [0.0, 1.0, 1.1, 2.0, 3.0],
                    [4.0, 5.0, 1.1, 6.0, 7.0],
                    [8.0, 9.0, 1.1, 10.0, 11.0],
                ]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, front_edge_columns=(0, 1), trails_columns=(0, 2)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 1.1, 2.0, 0.0],
                        [4.0, 0.0, 1.1, 6.0, 0.0],
                        [8.0, 0.0, 1.1, 10.0, 0.0],
                    ]
                )
            ).all()

        def test__front_edge_and_trails__2_regions_1_column_of_each__new_frame_is_edge_and_trail(
            self
        ):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 3, 0, 1), (0, 3, 3, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 1.1, 2.0, 3.0],
                    [4.0, 5.0, 1.1, 6.0, 7.0],
                    [8.0, 9.0, 1.1, 10.0, 11.0],
                ]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            new_frame = frame.serial_edges_and_trails_frame_from_frame(
                image, front_edge_columns=(0, 1), trails_columns=(0, 1)
            )

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 1.0, 0.0, 2.0, 3.0],
                        [4.0, 5.0, 0.0, 6.0, 7.0],
                        [8.0, 9.0, 0.0, 10.0, 11.0],
                    ]
                )
            ).all()

    class TestSerialAllTrailsArrayFromFrame:
        def test__left_quadrant__1_ci_region__1_serial_trail__extracts_all_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 2)])

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 3))

            image = np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_all_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 2.0],
                        [0.0, 0.0, 5.0],
                        [0.0, 0.0, 8.0],
                        [0.0, 0.0, 11.0],
                    ]
                )
            ).all()

        def test__left_quadrant__1_ci_region__2_serial_trail__extracts_all_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 2)])

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 4))

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 0.5],
                    [3.0, 4.0, 5.0, 0.5],
                    [6.0, 7.0, 8.0, 0.5],
                    [9.0, 10.0, 11.0, 0.5],
                ]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_all_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 2.0, 0.5],
                        [0.0, 0.0, 5.0, 0.5],
                        [0.0, 0.0, 8.0, 0.5],
                        [0.0, 0.0, 11.0, 0.5],
                    ]
                )
            ).all()

        def test__left_quadrant__2_ci_regions__2_serial_trail__extracts_all_trails(
            self
        ):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 0, 2), (2, 3, 0, 2)]
            )

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 4))

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 0.5],
                    [3.0, 4.0, 5.0, 0.5],
                    [6.0, 7.0, 8.0, 0.5],
                    [9.0, 10.0, 11.0, 0.5],
                ]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_all_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 2.0, 0.5],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 8.0, 0.5],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

        def test__same_as_above_but_right_quadrant__flips_trails_side(self):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 1, 2, 4), (2, 3, 2, 4)]
            )

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 0, 2))

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 0.5],
                    [3.0, 4.0, 5.0, 0.5],
                    [6.0, 7.0, 8.0, 0.5],
                    [9.0, 10.0, 11.0, 0.5],
                ]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_all_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [6.0, 7.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

    class TestSerialOverScanNonTrailsFromFrame:
        def test__left_quadrant__1_ci_region__serial_trails_go_over_2_right_hand_columns__2_pixels_above_kept(
            self
        ):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 3, 1, 2)])

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 3, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 3, 2, 4))

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_overscan_above_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
                )
            ).all()

        def test__left_quadrant__1_ci_region__serial_trails_go_over_1_right_hand_column__1_pixel_above_kept(
            self
        ):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 3, 1, 3)])

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 3, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 3, 3, 4))

            image = np.array(
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_overscan_above_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
                )
            ).all()

        def test__left_quadrant__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
            self
        ):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)]
            )

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 5, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 5, 3, 4))

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                ]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_overscan_above_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.0, 0.0, 0.0, 3.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 11.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 19.0],
                    ]
                )
            ).all()

        def test__right_quadrant__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
            self
        ):
            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)]
            )

            frame_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            frame_geometry.serial_prescan = ci_frame.Region((0, 5, 3, 4))
            frame_geometry.serial_overscan = ci_frame.Region((0, 5, 0, 1))

            image = np.array(
                [
                    [0.5, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                ]
            )

            frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=pattern)

            new_frame = frame.serial_overscan_above_trails_frame_from_frame(image)

            assert (
                new_frame
                == np.array(
                    [
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [8.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [16.0, 0.0, 0.0, 0.0],
                    ]
                )
            ).all()

    class TestSerialCalibrationArrayFromFrame:
        def test__geometry_left__ci_region_across_all_image__column_0__extracts_all_columns(
            self
        ):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 0, 5)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(0, 3))

            assert (
                serial_frame[0]
                == np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ]
                )
            ).all()

        def test__geometry_left__2_ci_regions__both_extracted(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 4.0, 4.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(0, 1))

            assert (
                serial_frame
                == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
            ).all()

        def test__geometry_right__2_ci_regions__both_extracted(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 4.0, 4.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(0, 1))

            assert (
                serial_frame
                == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
            ).all()

        def test__geometry_left__rows_cuts_out_bottom_row(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 0, 5)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(0, 2))

            assert (
                serial_frame
                == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
            ).all()

        def test__extract_two_regions_and_cut_bottom_row_from_each(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 4.0, 4.0, 4.0],
                    [0.0, 1.0, 5.0, 5.0, 5.0],
                    [0.0, 1.0, 6.0, 6.0, 6.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(0, 1))

            assert (
                serial_frame
                == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 5.0, 5.0, 5.0]])
            ).all()

        def test__extract_two_regions_and_cut_top_row_from_each(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 4.0, 4.0, 4.0],
                    [0.0, 1.0, 5.0, 5.0, 5.0],
                    [0.0, 1.0, 6.0, 6.0, 6.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            serial_frame = frame.serial_calibration_section_for_rows(image, rows=(1, 2))

            assert (
                serial_frame
                == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
            ).all()

    class TestSerialCalibrationSubArrays:
        def test__geometry_left__ci_region_across_all_image__column_0__extracts_all_columns(
            self
        ):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 0, 5)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_region = frame.serial_calibration_sub_arrays_from_frame(array=image)

            assert (
                serial_region[0]
                == np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ]
                )
            ).all()

        def test__geometry_left__ci_region_misses_serial_overscan__column_0__extracts_all_columns(
            self
        ):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 0, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_region = frame.serial_calibration_sub_arrays_from_frame(array=image)

            assert (
                serial_region[0]
                == np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ]
                )
            ).all()

        def test__geometry_left__ci_region_also_has_prescan__extracts_all_but_1_column(
            self
        ):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_region = frame.serial_calibration_sub_arrays_from_frame(array=image)

            assert (
                serial_region[0]
                == np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ]
                )
            ).all()

        def test__geometry_right__ci_region_has_prescan_and_overscan(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            serial_region = frame.serial_calibration_sub_arrays_from_frame(array=image)

            assert (
                serial_region[0]
                == np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                    ]
                )
            ).all()

        def test__geometry_left__2_ci_regions__both_extracted(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 3.0, 3.0, 3.0],
                    [0.0, 1.0, 4.0, 4.0, 4.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            serial_region = frame.serial_calibration_sub_arrays_from_frame(array=image)

            assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
            assert (serial_region[1] == np.array([[0.0, 1.0, 4.0, 4.0, 4.0]])).all()

    class TestParallelFrontEdgeFromFrame:
        def test__pattern_bottom___extracts_1_front_edge_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edge = frame.parallel_front_edge_arrays_from_frame(image, rows=(0, 1))
            assert (front_edge[0] == np.array([[1.0, 1.0, 1.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(image, rows=(1, 2))
            assert (front_edge[0] == np.array([[2.0, 2.0, 2.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(image, rows=(2, 3))
            assert (front_edge[0] == np.array([[3.0, 3.0, 3.0]])).all()

        def test__pattern_bottom___extracts_multiple_front_edges_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 5, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edge = frame.parallel_front_edge_arrays_from_frame(image, rows=(0, 2))
            assert (front_edge[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(image, rows=(1, 4))
            assert (
                front_edge[0]
                == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
            ).all()

        def test__pattern_bottom__2_regions__extracts_rows_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [6.0, 6.0, 6.0],  # <- Next front edge row.
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(0, 1)
            )
            assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
            assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(1, 2)
            )
            assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(2, 3)
            )
            assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(0, 3)
            )
            assert (
                front_edges[0]
                == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
            ).all()
            assert (
                front_edges[1]
                == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
            ).all()

        def test__pattern_top__does_all_the_above_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 5, 0, 3), (6, 9, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],  # <- Next front edge row.
                    [
                        4.0,
                        4.0,
                        4.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],  # <- Next front edge row.
                    [
                        8.0,
                        8.0,
                        8.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(0, 1)
            )
            assert (front_edges[0] == np.array([[4.0, 4.0, 4.0]])).all()
            assert (front_edges[1] == np.array([[8.0, 8.0, 8.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(1, 2)
            )
            assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(2, 3)
            )
            assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(0, 3)
            )
            assert (
                front_edges[0]
                == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
            ).all()
            assert (
                front_edges[1]
                == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])
            ).all()

        def test__mask_is_input__extracted_mask_and_masked_array_are_given(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [6.0, 6.0, 6.0],  # <- Next front edge row.
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            mask = np.array(
                [
                    [False, False, False],
                    [
                        False,
                        False,
                        False,
                    ],  # <- Front edge according to region and this frame_geometry
                    [False, True, False],  # <- Next front edge row.
                    [False, False, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.parallel_front_edge_arrays_from_frame(
                image, rows=(0, 3), mask=mask
            )
            assert (
                front_edges[0]
                == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
            ).all()
            assert (
                front_edges[0].mask
                == np.array(
                    [[False, False, False], [False, True, False], [False, False, True]]
                )
            ).all()

            assert (
                front_edges[1]
                == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
            ).all()
            assert (
                front_edges[1].mask
                == np.array(
                    [[False, False, False], [False, False, False], [True, False, False]]
                )
            ).all()

        def test__stacked_array_and_binned_line__2_regions__no_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [6.0, 6.0, 6.0],  # <- Next front edge row.
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # First fronrt edge array:
            #
            # [[1.0, 1.0, 1.0],
            #  [2.0, 2.0, 2.0],
            #  [3.0, 3.0, 3.0]])

            # Second front edge array:

            # [[5.0, 5.0, 5.0],
            #  [6.0, 6.0, 6.0],
            #  [7.0, 7.0, 7.0]]

            stacked_front_edges = frame.parallel_front_edge_stacked_array_from_frame(
                array=image, rows=(0, 3)
            )

            assert (
                stacked_front_edges
                == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
            ).all()

            front_edge_line = frame.parallel_front_edge_line_binned_over_columns_from_frame(
                array=image, rows=(0, 3)
            )

            assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        def test__same_as_above__include_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [6.0, 6.0, 6.0],  # <- Next front edge row.
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            mask = np.array(
                [
                    [False, False, False],
                    [
                        True,
                        False,
                        True,
                    ],  # <- Front edge according to region and this frame_geometry
                    [False, True, False],  # <- Next front edge row.
                    [False, False, True],
                    [False, False, False],
                    [
                        False,
                        False,
                        False,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # First fronrt edge array:
            #
            # [[1.0, 1.0, 1.0],
            #  [2.0, 2.0, 2.0],
            #  [3.0, 3.0, 3.0]])

            # Second front edge array:

            # [[5.0, 5.0, 5.0],
            #  [6.0, 6.0, 6.0],
            #  [7.0, 7.0, 7.0]]

            stacked_front_edges = frame.parallel_front_edge_stacked_array_from_frame(
                array=image, rows=(0, 3), mask=mask
            )

            assert (
                stacked_front_edges
                == np.ma.array([[5.0, 3.0, 5.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
            ).all()
            assert (
                stacked_front_edges.mask
                == np.ma.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                )
            ).all()

            front_edge_line = frame.parallel_front_edge_line_binned_over_columns_from_frame(
                array=image, rows=(0, 3), mask=mask
            )

            assert (front_edge_line == np.array([13.0 / 3.0, 14.0 / 3.0, 5.0])).all()

            mask = np.array(
                [
                    [False, False, False],
                    [
                        True,
                        False,
                        True,
                    ],  # <- Front edge according to region and this frame_geometry
                    [False, True, False],  # <- Next front edge row.
                    [False, False, True],
                    [False, False, False],
                    [
                        False,
                        False,
                        True,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # First fronrt edge array:
            #
            # [[1.0, 1.0, 1.0],
            #  [2.0, 2.0, 2.0],
            #  [3.0, 3.0, 3.0]])

            # Second front edge array:

            # [[5.0, 5.0, 5.0],
            #  [6.0, 6.0, 6.0],
            #  [7.0, 7.0, 7.0]]

            stacked_front_edges = frame.parallel_front_edge_stacked_array_from_frame(
                array=image, rows=(0, 3), mask=mask
            )

            assert (
                stacked_front_edges.mask
                == np.ma.array(
                    [[False, False, True], [False, False, False], [False, False, False]]
                )
            ).all()

        def test__no_rows_specified__uses_smallest_ci_pattern_rows(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3), (5, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [
                        1.0,
                        1.0,
                        1.0,
                    ],  # <- 1st Front edge according to region and this frame_geometry
                    [2.0, 2.0, 2.0],  # <- Next front edge row.
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [
                        5.0,
                        5.0,
                        5.0,
                    ],  # <- 2nd Front edge according to region and this frame_geometry
                    [6.0, 6.0, 6.0],  # <- Next front edge row.
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.parallel_front_edge_arrays_from_frame(image)
            assert (
                front_edges[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
            ).all()

            assert (
                front_edges[1] == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
            ).all()

            stacked_front_edges = frame.parallel_front_edge_stacked_array_from_frame(
                array=image
            )

            assert (
                stacked_front_edges == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
            ).all()

            front_edge_line = frame.parallel_front_edge_line_binned_over_columns_from_frame(
                array=image
            )

            assert (front_edge_line == np.array([3.0, 4.0])).all()

    class TestParallelTrailsFromFrame:
        def test__pattern_bottom__extracts_1_trails_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    # <- Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 1))

            assert (trails == np.array([[3.0, 3.0, 3.0]])).all()
            trails = frame.parallel_trails_arrays_from_frame(image, rows=(1, 2))

            assert (trails == np.array([[4.0, 4.0, 4.0]])).all()
            trails = frame.parallel_trails_arrays_from_frame(image, rows=(2, 3))
            assert (trails == np.array([[5.0, 5.0, 5.0]])).all()

        def test__pattern_bottom__extracts_multiple_trails_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    # <- Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )
            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 2))
            assert (trails == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(1, 3))
            assert (trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(1, 4))
            assert (
                trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
            ).all()

        def test__pattern_bottom__2_regions__extracts_rows_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    # <- 1st Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [7.0, 7.0, 7.0],  # <- Next trail.
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 1))
            assert (trails[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 2))
            assert (trails[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(1, 4))
            assert (
                trails[0]
                == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
            ).all()
            assert (
                trails[1]
                == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
            ).all()

        def test__pattern_top__does_all_the_above_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(3, 5, 0, 3), (9, 10, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],  # <- Next trail.
                    [2.0, 2.0, 2.0],
                    # <- 1st Trails form here onwards according to region and this frame_geometry
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],  # <- Next trail.
                    [8.0, 8.0, 8.0],
                    # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 1))
            assert (trails[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (trails[1] == np.array([[8.0, 8.0, 8.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(0, 2))
            assert (trails[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()
            assert (trails[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(image, rows=(1, 3))
            assert (trails[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        def test__mask_is_input__extracted_mask_and_masked_array_are_given(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [
                        3.0,
                        3.0,
                        3.0,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [
                        6.0,
                        6.0,
                        6.0,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [7.0, 7.0, 7.0],  # <- Next trail.
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            mask = np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                    [
                        False,
                        True,
                        True,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [False, False, False],  # <- Next trail.
                    [False, False, False],
                    [
                        False,
                        False,
                        False,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [True, False, False],  # <- Next trail.
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(
                image, rows=(0, 2), mask=mask
            )
            assert (trails[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
            assert (
                trails[0].mask == np.array([[False, True, True], [False, False, False]])
            ).all()

            assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()
            assert (
                trails[1].mask
                == np.array([[False, False, False], [True, False, False]])
            ).all()

        def test__stacked_array_and_binned_line__2_regions__no_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [
                        3.0,
                        3.0,
                        3.0,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [
                        6.0,
                        6.0,
                        6.0,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [7.0, 7.0, 7.0],  # <- Next trail.
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # [3.0, 3.0, 3.0],
            #  [4.0, 4.0, 4.0]]

            # Array 2:

            # [[6.0, 6.0, 6.0],
            # [7.0, 7.0, 7.0]]

            stacked_trails = frame.parallel_trails_stacked_array_from_frame(
                image, rows=(0, 2)
            )

            assert (
                stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])
            ).all()

            trails_line = frame.parallel_trails_line_binned_over_columns_from_frame(
                array=image, rows=(0, 2)
            )

            assert (trails_line == np.array([4.5, 5.5])).all()

        def test__same_as_above__include_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [
                        3.0,
                        3.0,
                        3.0,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [4.0, 4.0, 4.0],  # <- Next trail.
                    [5.0, 5.0, 5.0],
                    [
                        6.0,
                        6.0,
                        6.0,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [7.0, 7.0, 7.0],  # <- Next trail.
                    [8.0, 8.0, 8.0],
                    [9.0, 9.0, 9.0],
                ]
            )

            mask = np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                    [
                        False,
                        True,
                        True,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [False, False, False],  # <- Next trail.
                    [False, False, False],
                    [
                        False,
                        False,
                        False,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [True, False, False],  # <- Next trail.
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # [3.0, 3.0, 3.0],
            #  [4.0, 4.0, 4.0]]

            # Array 2:

            # [[6.0, 6.0, 6.0],
            # [7.0, 7.0, 7.0]]

            stacked_trails = frame.parallel_trails_stacked_array_from_frame(
                image, rows=(0, 2), mask=mask
            )

            assert (
                stacked_trails == np.array([[4.5, 6.0, 6.0], [4.0, 5.5, 5.5]])
            ).all()
            assert (
                stacked_trails.mask
                == np.array([[False, False, False], [False, False, False]])
            ).all()

            trails_line = frame.parallel_trails_line_binned_over_columns_from_frame(
                array=image, rows=(0, 2), mask=mask
            )

            assert (trails_line == np.array([16.5 / 3.0, 15.0 / 3.0])).all()

            mask = np.array(
                [
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                    [
                        False,
                        True,
                        True,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [False, False, False],  # <- Next trail.
                    [False, False, False],
                    [
                        False,
                        False,
                        True,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [True, False, False],  # <- Next trail.
                    [False, False, False],
                    [False, False, False],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            stacked_trails = frame.parallel_trails_stacked_array_from_frame(
                image, rows=(0, 2), mask=mask
            )

            assert (
                stacked_trails.mask
                == np.array([[False, False, True], [False, False, False]])
            ).all()

        def test__no_rows_specified__uses_smallest_parallel_trails_size(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(1, 4, 0, 3), (6, 8, 0, 3)]
            )

            image = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0],
                    [
                        4.0,
                        4.0,
                        4.0,
                    ],  # <- 1st Trails form here onwards according to region and this frame_geometry
                    [5.0, 5.0, 5.0],  # <- Next trail
                    [6.0, 6.0, 6.0],
                    [7.0, 7.0, 7.0],
                    [
                        8.0,
                        8.0,
                        8.0,
                    ],  # <- 2nd Trails form here onwards according to region and this frame_geometry
                    [9.0, 9.0, 9.0],
                ]
            )  # 2nd Trail starts here

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.parallel_trails_arrays_from_frame(image)
            assert (trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()
            assert (trails[1] == np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])).all()

            stacked_trails = frame.parallel_trails_stacked_array_from_frame(image)

            assert (
                stacked_trails == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
            ).all()

            trails_line = frame.parallel_trails_line_binned_over_columns_from_frame(
                array=image
            )

            assert (trails_line == np.array([6.0, 7.0])).all()

    class TestSerialFrontEdgeFromFrame:
        def test__pattern_bottom___extracts_1_front_edge_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #       /| Front Edge

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edge = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 1)
            )

            assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(
                image, columns=(1, 2)
            )

            assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(
                image, columns=(2, 3)
            )

            assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        def test__pattern_bottom___extracts_multiple_front_edges_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 5)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                    /| Front Edge

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edge = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 2)
            )

            assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(
                image, columns=(1, 4)
            )

            assert (
                front_edge
                == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
            ).all()

        def test__pattern_bottom__2_regions__extracts_columns_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                    /| FE 1        /\ FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 1)
            )

            assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
            assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(1, 2)
            )

            assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
            assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(2, 3)
            )

            assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
            assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 3)
            )

            assert (
                front_edges[0]
                == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            ).all()

            assert (
                front_edges[1]
                == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
            ).all()

        def test__pattern_right__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                               /| FE 1            /\ FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 1)
            )

            assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
            assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(1, 2)
            )

            assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
            assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(2, 3)
            )

            assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
            assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 3)
            )

            assert (
                front_edges[0]
                == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            ).all()

            assert (
                front_edges[1]
                == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
            ).all()

        def test__mask_is_input__extracted_mask_and_masked_array_are_given(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                    /| FE 1        /\ FE 2

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                    ],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.serial_front_edge_arrays_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (
                front_edges[0]
                == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            ).all()
            assert (
                (front_edges[0].mask)
                == np.array(
                    [[False, False, False], [False, True, False], [False, False, False]]
                )
            ).all()

            assert (
                front_edges[1]
                == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
            ).all()
            assert (
                front_edges[1].mask
                == np.array(
                    [[False, False, False], [False, True, False], [False, False, True]]
                )
            ).all()

        def test__stacked_array_and_binned_line__2_regions__no_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                      /| FE 1                /\ FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            stacked_front_edges = frame.serial_front_edge_stacked_array_from_frame(
                image, columns=(0, 3)
            )

            # [[1.0, 2.0, 3.0],
            #  [1.0, 2.0, 3.0],
            #  [1.0, 2.0, 3.0]]

            # [[5.0, 6.0, 7.0],
            #  [5.0, 6.0, 7.0],
            #  [5.0, 6.0, 7.0]]

            assert (
                stacked_front_edges
                == np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
            ).all()

            front_edge_line = frame.serial_front_edge_line_binned_over_rows_from_frame(
                image, columns=(0, 3)
            )

            assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        def test__same_as_above__include_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                      /| FE 1                /\ FE 2

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                        True,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                    ],
                ]
            )

            #                        /| FE 1                       /| FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            stacked_front_edges = frame.serial_front_edge_stacked_array_from_frame(
                image, columns=(0, 3), mask=mask
            )

            # [[1.0, 2.0, 3.0],
            #  [1.0, 2.0, 3.0],
            #  [1.0, 2.0, 3.0]]

            # [[5.0, 6.0, 7.0],
            #  [5.0, 6.0, 7.0],
            #  [5.0, 6.0, 7.0]]

            assert (
                stacked_front_edges
                == np.array([[3.0, 4.0, 5.0], [3.0, 2.0, 7.0], [3.0, 4.0, 3.0]])
            ).all()
            assert (
                stacked_front_edges.mask
                == np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                )
            ).all()

            front_edge_line = frame.serial_front_edge_line_binned_over_rows_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (front_edge_line == np.array([3.0, 10.0 / 3.0, 5.0])).all()

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [False, False, False, True, False, False, True, True, False, False],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        False,
                    ],
                ]
            )

            #                        /| FE 1                       /| FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            stacked_front_edges = frame.serial_front_edge_stacked_array_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (
                stacked_front_edges.mask
                == np.array(
                    [[False, False, False], [False, False, True], [False, False, False]]
                )
            ).all()

        def test__no_columns_specified_so_uses_smallest_charge_injection_region_column_size(
            self
        ):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 3), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                    /| FE 1        /\ FE 2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            front_edges = frame.serial_front_edge_arrays_from_frame(image)

            assert (
                front_edges[0] == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
            ).all()

            assert (
                front_edges[1] == np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])
            ).all()

            stacked_front_edges = frame.serial_front_edge_stacked_array_from_frame(
                image
            )

            assert (
                stacked_front_edges == np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
            ).all()

            front_edge_line = frame.serial_front_edge_line_binned_over_rows_from_frame(
                image
            )

            assert (front_edge_line == np.array([3.0, 4.0])).all()

    class TestSerialTrailsFromFrame:
        def test__pattern_bottom___extracts_1_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                                    /| Trails Begin

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 1))

            assert (trails == np.array([[4.0], [4.0], [4.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(1, 2))

            assert (trails == np.array([[5.0], [5.0], [5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(2, 3))

            assert (trails == np.array([[6.0], [6.0], [6.0]])).all()

        def test__pattern_bottom___extracts_multiple_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ]
            )

            #                                   /| Trails Begin

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 2))

            assert (trails == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(1, 4))

            assert (
                trails == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
            ).all()

        def test__pattern_bottom__2_regions__extracts_columns_correctly(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #                                   /| Trails1           /\ Trails2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 1))

            assert (trails[0] == np.array([[4.0], [4.0], [4.0]])).all()
            assert (trails[1] == np.array([[8.0], [8.0], [8.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(1, 2))

            assert (trails[0] == np.array([[5.0], [5.0], [5.0]])).all()
            assert (trails[1] == np.array([[9.0], [9.0], [9.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(2, 3))

            assert (trails[0] == np.array([[6.0], [6.0], [6.0]])).all()
            assert (trails[1] == np.array([[10.0], [10.0], [10.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 3))

            assert (
                trails[0]
                == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
            ).all()

            assert (
                trails[1]
                == np.array([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
            ).all()

        def test__pattern_right__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 3, 6), (0, 3, 8, 11)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #               Trails1   /|                Trails2 /\

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                ci_pattern=pattern,
            )

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 1))

            assert (trails[0] == np.array([[2.0], [2.0], [2.0]])).all()
            assert (trails[1] == np.array([[7.0], [7.0], [7.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(1, 2))

            assert (trails[0] == np.array([[1.0], [1.0], [1.0]])).all()
            assert (trails[1] == np.array([[6.0], [6.0], [6.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(2, 3))

            assert (trails[0] == np.array([[0.0], [0.0], [0.0]])).all()
            assert (trails[1] == np.array([[5.0], [5.0], [5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(image, columns=(0, 3))

            assert (
                trails[0]
                == np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
            ).all()

            assert (
                trails[1]
                == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
            ).all()

        def test__mask_is_input__extracted_mask_and_masked_array_are_given(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #                                   /| Trails1           /\ Trails2

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                    ],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            trails = frame.serial_trails_arrays_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (
                trails[0]
                == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
            ).all()
            assert (
                trails[0].mask
                == np.array(
                    [[False, False, False], [True, False, True], [False, False, False]]
                )
            ).all()

            assert (
                trails[1]
                == np.array([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
            ).all()
            assert (
                trails[1].mask
                == np.array(
                    [[False, False, False], [False, False, False], [False, True, False]]
                )
            ).all()

        def test__stacked_array_and_binned_line__2_regions__no_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #                                   /| Trails1           /\ Trails2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # [[4.0, 5.0, 6.0],
            #  [4.0, 5.0, 6.0],
            #  [4.0, 5.0, 6.0]]

            # [[8.0, 9.0, 10.0],
            #  [8.0, 9.0, 10.0],
            #  [8.0, 9.0, 10.0]]

            stacked_trails = frame.serial_trails_stacked_array_from_frame(
                image, columns=(0, 3)
            )

            assert (
                stacked_trails
                == np.array([[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 7.0, 8.0]])
            ).all()

            trails_line = frame.serial_trails_line_binned_over_rows_from_frame(
                image, columns=(0, 3)
            )

            assert (trails_line == np.array([6.0, 7.0, 8.0])).all()

        def test__same_as_above__include_masking(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #                                      /| Trails1           /\ Trails2

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                    ],
                ]
            )

            #                                               /| Trails1                   /\ Trails2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            # [[4.0, 5.0, 6.0],
            #  [4.0, 5.0, 6.0],
            #  [4.0, 5.0, 6.0]]

            # [[8.0, 9.0, 10.0],
            #  [8.0, 9.0, 10.0],
            #  [8.0, 9.0, 10.0]]

            stacked_trails = frame.serial_trails_stacked_array_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (
                stacked_trails
                == np.array([[6.0, 7.0, 8.0], [8.0, 7.0, 10.0], [6.0, 5.0, 8.0]])
            ).all()
            assert (
                stacked_trails.mask
                == np.array(
                    [
                        [False, False, False],
                        [False, False, False],
                        [False, False, False],
                    ]
                )
            ).all()

            trails_line = frame.serial_trails_line_binned_over_rows_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (trails_line == np.array([20.0 / 3.0, 19.0 / 3.0, 26.0 / 3.0])).all()

            mask = np.array(
                [
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                    ],
                    [
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                        False,
                        True,
                        False,
                    ],
                ]
            )

            #                                               /| Trails1                   /\ Trails2

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            stacked_trails = frame.serial_trails_stacked_array_from_frame(
                image, columns=(0, 3), mask=mask
            )

            assert (
                stacked_trails.mask
                == np.array(
                    [[False, False, False], [False, False, False], [False, True, False]]
                )
            ).all()

        def test__no_columns_specified_so_uses_full_serial_overscan(self):

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 4, 7)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                ]
            )

            #                                      /| Trails1           /\ Trails2

            ci_geometry = ci_frame.FrameGeometry(
                serial_overscan=ci_frame.Region((0, 1, 0, 4)),
                serial_prescan=ci_frame.Region((0, 1, 0, 1)),
                parallel_overscan=ci_frame.Region((0, 1, 0, 1)),
                corner=(0, 0),
            )

            frame = ci_frame.ChInj(frame_geometry=ci_geometry, ci_pattern=pattern)

            trails = frame.serial_trails_arrays_from_frame(image)

            assert (
                trails[0]
                == np.array(
                    [[4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]]
                )
            ).all()

            assert (
                trails[1]
                == np.array(
                    [
                        [7.0, 8.0, 9.0, 10.0],
                        [7.0, 8.0, 9.0, 10.0],
                        [7.0, 8.0, 9.0, 10.0],
                    ]
                )
            ).all()

            stacked_trails = frame.serial_trails_stacked_array_from_frame(image)

            assert (
                stacked_trails
                == np.array(
                    [[5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5]]
                )
            ).all()

            trails_line = frame.serial_trails_line_binned_over_rows_from_frame(image)

            assert (trails_line == np.array([5.5, 6.5, 7.5, 8.5])).all()

    class TestParallelSerialCalibrationSection:
        def test__extracts_everything_except_prescan(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 1, 0, 1)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            frame.frame_geometry.serial_prescan = ci_frame.Region(region=(0, 4, 0, 1))

            extracted_array = frame.parallel_serial_calibration_section(image)

            assert (
                extracted_array
                == np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                    ]
                )
            ).all()

        def test__extracts_everything_except_prescan_2(self):
            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[(0, 1, 0, 1)]
            )

            image = np.array(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                    [0.0, 1.0, 2.0, 3.0],
                ]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            frame.frame_geometry.serial_prescan = ci_frame.Region(region=(0, 4, 0, 2))

            extracted_array = frame.parallel_serial_calibration_section(image)

            assert (
                extracted_array
                == np.array(
                    [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
                )
            ).all()

    class TestSmallestPArallelTrailsRows:
        def test__x1_ci_region__bottom_frame_geometry(self):

            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(5, 5)
            )

            assert smallest_parallel_trails_rows == 2

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(7, 5)
            )

            assert smallest_parallel_trails_rows == 4

        def test__x2_ci_region__bottom_frame_geometry(self):

            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(0, 3, 0, 3), (5, 7, 0, 3)]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                ci_pattern=pattern,
            )

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(10, 5)
            )

            assert smallest_parallel_trails_rows == 2

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(8, 5)
            )

            assert smallest_parallel_trails_rows == 1

        def test__x2_ci_region__top_frame_geometry(self):

            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(1, 4, 0, 3), (5, 7, 0, 3)]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                ci_pattern=pattern,
            )

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(10, 5)
            )

            assert smallest_parallel_trails_rows == 1

            pattern = ci_pattern.CIPattern(
                normalization=10.0, regions=[(8, 12, 0, 3), (14, 16, 0, 3)]
            )

            frame = ci_frame.ChInj(
                frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                ci_pattern=pattern,
            )

            smallest_parallel_trails_rows = frame.smallest_parallel_trails_rows_from_shape(
                shape=(10, 5)
            )

            assert smallest_parallel_trails_rows == 2

    class TestSerialTrailsColumns:
        def test__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
            self
        ):

            ci_geometry = ci_frame.FrameGeometry(
                serial_overscan=ci_frame.Region((0, 1, 0, 10)),
                serial_prescan=ci_frame.Region((0, 1, 0, 1)),
                parallel_overscan=ci_frame.Region((0, 1, 0, 1)),
                corner=(0, 0),
            )

            assert ci_geometry.serial_trails_columns == 10

            ci_geometry = ci_frame.FrameGeometry(
                serial_overscan=ci_frame.Region((0, 1, 0, 50)),
                serial_prescan=ci_frame.Region((0, 1, 0, 1)),
                parallel_overscan=ci_frame.Region((0, 1, 0, 1)),
                corner=(0, 0),
            )

            assert ci_geometry.serial_trails_columns == 50


class TestQuadGeometryEuclid_bottom_left(object):
    class TestParallelFrontEdgeRegion:
        def test__extract_one_row__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is row 0, so for 1 row we extract 0 -> 1

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(0, 1)
            )

            assert ci_front_edge == (0, 1, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is row 0, so for 2 rows we extract 0 -> 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(0, 2)
            )

            assert ci_front_edge == (0, 2, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is row 0, so for these 2 rows we extract 1 ->2

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_front_edge == (1, 3, 0, 3)

    class TestParallelTrailsRegion:
        def test__extract_one_row__takes_coordinates_after_bottom_of_region(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # The trails are row 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(0, 1)
            )

            assert ci_trails == (3, 4, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are row 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(0, 2)
            )

            assert ci_trails == (3, 5, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are row 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_trails == (4, 6, 0, 3)

        def test__parallel_trail_size_to_image_edge(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[ci_frame.Region(region=(0, 3, 0, 3))]
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(5, 100)
            )

            assert parallel_trail_size == 2

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(7, 100)
            )

            assert parallel_trail_size == 4

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0,
                regions=[
                    ci_frame.Region(region=(0, 2, 0, 3)),
                    ci_frame.Region(region=(5, 8, 0, 3)),
                    ci_frame.Region(region=(11, 14, 0, 3)),
                ],
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(15, 100)
            )

            assert parallel_trail_size == 1

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(20, 100)
            )

            assert parallel_trail_size == 6

    class TestParallelSideNearestReadOut:
        def test__columns_0_to_1__region_is_left_hand_side(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(5, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 5, 0, 1)

        def test__columns_1_to_3__region_is_left_hand_side(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(4, 4), columns=(1, 3)
            )

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_1_to_3__different_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 2, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(4, 4), columns=(1, 3)
            )

            assert ci_parallel_region == (0, 4, 3, 5)

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(2, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 2, 0, 1)

    class TestSerialFrontEdgeRegion:
        def test__extract_one_column__takes_coordinates_from_left_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for 1 column we extract 0 -> 1

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(0, 1)
            )

            assert ci_front_edge == (0, 3, 0, 1)

        def test__extract_two_columns__first_and_second__takes_coordinates_from_left_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for 2 columns we extract 0 -> 2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(0, 2)
            )

            assert ci_front_edge == (0, 3, 0, 2)

        def test__extract_two_columns__second_and_third__takes_coordinates_from_left_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_front_edge == (0, 3, 1, 3)

    class TestSerialTrailsRegion:
        def test__extract_one_row__takes_coordinates_after_right_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # The trails are column 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(0, 1)
            )

            assert ci_trails == (0, 3, 3, 4)

        def test__extract_two_columns__first_and_second__takes_coordinates_after_right_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are column 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(0, 2)
            )

            assert ci_trails == (0, 3, 3, 5)

        def test__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_trails == (0, 3, 4, 6)

    class TestSerialChargeInjectionAndTrails:
        def test__column_0_of_front_edge__region_is_left_hand_side__no_overscan_beyond_ci_region(
            self
        ):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(5, 5)
            )

            assert ci_serial_region == (1, 3, 0, 5)

        def test__column_0_of_front_edge__region_is_left_hand_side__overscan_beyond_ci_region(
            self
        ):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(5, 25)
            )

            assert ci_serial_region == (1, 3, 0, 25)


class TestQuadGeometryEuclid_bottom_right(object):
    class TestParallelFrontEdgeOfRegion:
        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is row 0, so for these 2 rows we extract 1 ->2

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_front_edge == (1, 3, 0, 3)

    class TestParallelTrailsRegion:
        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are row 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_trails == (4, 6, 0, 3)

        def test__parallel_trail_size_to_image_edge(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[ci_frame.Region(region=(0, 3, 0, 3))]
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(5, 100)
            )

            assert parallel_trail_size == 2

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(7, 100)
            )

            assert parallel_trail_size == 4

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0,
                regions=[
                    ci_frame.Region(region=(0, 2, 0, 3)),
                    ci_frame.Region(region=(5, 8, 0, 3)),
                    ci_frame.Region(region=(11, 14, 0, 3)),
                ],
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(15, 100)
            )

            assert parallel_trail_size == 1

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(20, 100)
            )

            assert parallel_trail_size == 6

    class TestParallelSideNearestReadOut:
        def test__columns_0_to_1__region_is_right_hand_side(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(5, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 5, 4, 5)

        def test__columns_1_to_3__region_is_right_hand_side(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 0, 4))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(4, 4), columns=(1, 3)
            )

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_1_to_3__different_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 2, 4))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(4, 4), columns=(1, 3)
            )

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(0, 1, 0, 5))
            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(2, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 2, 4, 5)

    class TestSerialFrontEdgeRegion:
        def test__extract_one_column__takes_coordinates_from_right_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for 1 column we extract 0 -> 1

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(0, 1)
            )

            assert ci_front_edge == (0, 3, 2, 3)

        def test__extract_two_columns__first_and_second__takes_coordinates_from_right_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for 2 columns we extract 0 -> 2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(0, 2)
            )

            assert ci_front_edge == (0, 3, 1, 3)

        def test__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_front_edge == (0, 3, 0, 2)

    class TestSerialTrailsRegion:
        def test__extract_one_row__takes_coordinates_after_left_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 3, 6)
            )  # The trails are column 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(0, 1)
            )

            assert ci_trails == (0, 3, 2, 3)

        def test__extract_two_columns__first_and_second__takes_coordinates_after_left_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                (0, 3, 3, 6)
            )  # The trails are column 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(0, 2)
            )

            assert ci_trails == (0, 3, 1, 3)

        def test__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                (0, 3, 3, 6)
            )  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_trails == (0, 3, 0, 2)

    class TestSerialChargeInjectionAndTrails:
        def test__front_edge__region_is_left_hand_side__no_overscan_beyond_ci_region(
            self
        ):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(5, 5)
            )

            assert ci_serial_region == (1, 3, 0, 5)

        def test__front_edge__region_is_left_hand_side__overscan_beyond_ci_region(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 20, 25))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(5, 25)
            )

            assert ci_serial_region == (1, 3, 0, 25)

        def test__ci_region_has_overscan_and_prescan_either_side__prescan_included(
            self
        ):

            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()
            ci_region = ci_frame.Region(region=(1, 3, 10, 20))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(5, 25)
            )

            assert ci_serial_region == (1, 3, 0, 25)


class TestQuadGeometryEuclid_top_left(object):
    class TestParallelFrontEdgeOfRegion:
        def test__extract_one_row__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The front edge is closest to 3, so for 1 edge we extract row 3-> 4

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(0, 1)
            )

            assert ci_front_edge == (2, 3, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The front edge is closest to 3, so for these 2 rows we extract 2 & 3

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(0, 2)
            )

            assert ci_front_edge == (1, 3, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_front_edge == (0, 2, 0, 3)

    class TestParallelTrailsRegion:
        def test__extract_one_row__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (3, 5, 0, 3)
            )  # The trails are the rows after row 3, so for 1 edge we should extract just row 2

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(0, 1)
            )

            assert ci_trails == (2, 3, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (3, 5, 0, 3)
            )  # The trails are the row after row 3, so for these 2 edges we extract rows 1->3

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(0, 2)
            )

            assert ci_trails == (1, 3, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (3, 5, 0, 3)
            )  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_trails == (0, 2, 0, 3)

        def test__parallel_trail_size_to_image_edge(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[ci_frame.Region(region=(0, 3, 0, 3))]
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(5, 100)
            )

            assert parallel_trail_size == 0

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(7, 100)
            )

            assert parallel_trail_size == 0

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0,
                regions=[
                    ci_frame.Region(region=(5, 6, 0, 3)),
                    ci_frame.Region(region=(11, 12, 0, 3)),
                    ci_frame.Region(region=(17, 18, 0, 3)),
                ],
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(20, 100)
            )

            assert parallel_trail_size == 5

    class TestParallelSideNearestReadOut:
        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_left()
            ci_region = ci_frame.Region(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(2, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 2, 0, 1)

    class TestSerialFrontEdgeRegion:
        def test__extract_two_columns__second_and_third__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_front_edge == (0, 3, 1, 3)

    class TestSerialTrailsRegion:
        def test__extract_two_columns__second_and_third__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_trails == (0, 3, 4, 6)

    class TestSerialChargeInjectionAndTrails:
        def test__different_ci_region(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_top_left()
            ci_region = ci_frame.Region(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(8, 55)
            )

            assert ci_serial_region == (3, 5, 0, 55)


class TestQuadGeometryEuclid_top_right(object):
    class TestParallelFrontEdgeOfRegion:
        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_right()

            ci_region = ci_frame.Region(
                (0, 3, 0, 3)
            )  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_front_edge == (0, 2, 0, 3)

    class TestParallelTrailsRegion:
        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_right()

            ci_region = ci_frame.Region(
                (3, 5, 0, 3)
            )  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

            ci_trails = ci_geometry.parallel_trails_region(
                ci_region=ci_region, rows=(1, 3)
            )

            assert ci_trails == (0, 2, 0, 3)

        def test__parallel_trail_size_to_image_edge(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_top_right()

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0, regions=[ci_frame.Region(region=(0, 3, 0, 3))]
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(5, 100)
            )

            assert parallel_trail_size == 0

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(7, 100)
            )

            assert parallel_trail_size == 0

            pattern = ci_pattern.CIPatternUniform(
                normalization=1.0,
                regions=[
                    ci_frame.Region(region=(5, 6, 0, 3)),
                    ci_frame.Region(region=(11, 12, 0, 3)),
                    ci_frame.Region(region=(17, 18, 0, 3)),
                ],
            )

            parallel_trail_size = ci_geometry.parallel_trail_size_to_image_edge(
                ci_pattern=pattern, shape=(20, 100)
            )

            assert parallel_trail_size == 5

    class TestParallelSideNearestReadOut:
        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.FrameGeometry.euclid_top_right()
            ci_region = ci_frame.Region(region=(0, 1, 0, 5))
            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(
                ci_region=ci_region, image_shape=(2, 5), columns=(0, 1)
            )

            assert ci_parallel_region == (0, 2, 4, 5)

    class TestSerialFrontEdgeRegion:
        def test__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                region=(0, 3, 0, 3)
            )  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_front_edge == (0, 3, 0, 2)

    class TestSerialTrailsRegion:
        def test__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(
            self
        ):
            ci_geometry = ci_frame.FrameGeometry.euclid_bottom_right()

            ci_region = ci_frame.Region(
                (0, 3, 3, 6)
            )  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(
                ci_region=ci_region, columns=(1, 3)
            )

            assert ci_trails == (0, 3, 0, 2)

    class TestSerialChargeInjectionAndTrails:
        def test__different_ci_region(self):

            ci_geometry = ci_frame.FrameGeometry.euclid_top_right()
            ci_region = ci_frame.Region(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_prescan_ci_region_and_trails(
                ci_region=ci_region, image_shape=(8, 55)
            )

            assert ci_serial_region == (3, 5, 0, 55)