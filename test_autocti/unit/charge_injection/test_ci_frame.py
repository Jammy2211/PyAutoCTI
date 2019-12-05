import numpy as np

import autocti as ac
from autocti.structures import frame
from test_autocti.mock.mock import MockCIGeometry


class TestCiRegionsArray:
    def test__1_ci_region__extracted_correctly(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])
        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        assert (
            frame.ci_regions_array
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

    def test__2_ci_regions__extracted_correctly(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 1, 2), (2, 3, 1, 3)]
        )
        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        assert (
            frame.ci_regions_array
            == np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()


class TestNonCIRegionFrame:
    def test__1_ci_region__parallel_overscan_is_entire_image__extracts_everything_between_its_columns(
        self
    ):

        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(array=arr, ci_pattern=ci_pattern)

        assert (
            frame.non_ci_regions_frame
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
            )
        ).all()
        assert frame.non_ci_regions_frame.ci_pattern == ci_pattern

    def test__same_as_above_but_2_ci_regions(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, ci_pattern=ci_pattern)

        assert (
            frame.non_ci_regions_frame
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
        assert frame.non_ci_regions_frame.ci_pattern == ci_pattern


class TestParallelEdgesAndTrailsFrame:
    def test__front_edge_only__multiple_rows__new_frame_contains_only_edge(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(front_edge_rows=(0, 1))

        assert (
            new_frame
            == np.array(
                [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(front_edge_rows=(0, 2))

        assert (
            new_frame
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

    def test__trails_only__new_frame_contains_only_trails(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(trails_rows=(0, 1))

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

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(trails_rows=(0, 2))

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
        assert new_frame.ci_pattern == ci_pattern

    def test__front_edge_and_trails__2_rows_of_each__new_frame_is_edge_and_trail(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(
            front_edge_rows=(0, 2), trails_rows=(0, 2)
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
        assert new_frame.ci_pattern == ci_pattern

    def test__front_edge_and_trails__2_regions__1_row_of_each__new_frame_is_edge_and_trail(
        self
    ):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.parallel_edges_and_trails_frame(
            front_edge_rows=(0, 1), trails_rows=(0, 1)
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
        assert new_frame.ci_pattern == ci_pattern


class TestParallelCalibrationSection:
    def test__left__columns_0_to_1__extracts_1_column_left_hand_side_of_array(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        extracted_side = frame.parallel_calibration_section_for_columns(columns=(0, 1))

        assert (extracted_side == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()

    def test__bottom__columns_1_to_3__extracts_2_columns_middle_and_right_of_array(
        self
    ):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        extracted_side = frame.parallel_calibration_section_for_columns(columns=(1, 3))

        assert (
            extracted_side
            == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ).all()

    def test__right__columns_1_to_3__extracts_2_columns_middle_and_left_of_array(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        extracted_side = frame.parallel_calibration_section_for_columns(columns=(1, 3))

        assert (
            extracted_side
            == np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        ).all()


class TestSerialEdgesAndTrailsFrame:
    def test__front_edge_only__multiple_columns__new_frame_is_only_edge(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )
        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.serial_edges_and_trails_frame(front_edge_columns=(0, 1))

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
            )
        ).all()

        new_frame = frame.serial_edges_and_trails_frame(front_edge_columns=(0, 2))

        assert (
            new_frame
            == np.array(
                [[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

    def test__trails_only__multiple_columns__new_frame_is_only_that_trail(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )
        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.serial_edges_and_trails_frame(trails_columns=(0, 1))

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]]
            )
        ).all()

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.serial_edges_and_trails_frame(trails_columns=(0, 2))

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

    def test__front_edge_and_trails__2_columns_of_each__new_frame_is_edge_and_trail(
        self
    ):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 2)])

        arr = np.array(
            [
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ]
        )
        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 2)
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
        assert new_frame.ci_pattern == ci_pattern

    def test__front_edge_and_trails__2_regions_1_column_of_each__new_frame_is_edge_and_trail(
        self
    ):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 3, 0, 1), (0, 3, 3, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ]
        )
        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        new_frame = frame.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 1)
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
        assert new_frame.ci_pattern == ci_pattern


class TestSerialAllTrailsFrame:
    def test__left__1_ci_region__multiple_serial_trail__extracts_all_trails(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(1, 0),
            ci_pattern=ci_pattern,
            serial_overscan=(0, 4, 2, 3),
        )

        new_frame = frame.serial_trails_frame

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(1, 0),
            ci_pattern=ci_pattern,
            serial_overscan=(0, 4, 2, 4),
        )

        new_frame = frame.serial_trails_frame

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
        assert new_frame.ci_pattern == ci_pattern

    def test__left__2_ci_regions__2_serial_trail__extracts_all_trails(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 0, 2), (2, 3, 0, 2)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(1, 0),
            ci_pattern=ci_pattern,
            serial_overscan=(0, 4, 2, 4),
        )

        new_frame = frame.serial_trails_frame

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
        assert new_frame.ci_pattern == ci_pattern

    def test__same_as_above_but_right_quadrant__flips_trails_side(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 2, 4), (2, 3, 2, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(1, 1),
            ci_pattern=ci_pattern,
            serial_overscan=(0, 4, 0, 2),
        )

        new_frame = frame.serial_trails_frame

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
        assert new_frame.ci_pattern == ci_pattern


class TestSerialOverScanAboveTrailsFrame:
    def test__left__1_ci_region__serial_trails_go_over_hand_columns(self):
        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(1, 3, 1, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(0, 0),
            ci_pattern=ci_pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 2, 4),
        )

        new_frame = frame.serial_overscan_above_trails_frame

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(1, 3, 1, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(0, 0),
            ci_pattern=ci_pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        new_frame = frame.serial_overscan_above_trails_frame

        assert (
            new_frame
            == np.array(
                [[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_frame.ci_pattern == ci_pattern

    def test__left__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
        self
    ):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(0, 0),
            ci_pattern=ci_pattern,
            serial_prescan=(0, 5, 0, 1),
            serial_overscan=(0, 5, 3, 4),
        )

        new_frame = frame.serial_overscan_above_trails_frame

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
        assert new_frame.ci_pattern == ci_pattern

    def test__right__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
        self
    ):
        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)]
        )

        arr = np.array(
            [
                [0.5, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ]
        )

        frame = ac.ci_frame.manual(
            array=arr,
            corner=(1, 1),
            ci_pattern=ci_pattern,
            serial_prescan=(0, 5, 3, 4),
            serial_overscan=(0, 5, 0, 1),
        )

        new_frame = frame.serial_overscan_above_trails_frame

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
        assert new_frame.ci_pattern == ci_pattern


class TestSerialCalibrationSection:
    def test__left__ci_region_across_all_image__column_0__extracts_all_columns(self):

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(0, 3))

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

    def test__left__2_ci_regions__both_extracted(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
        ).all()

    def test__right__2_ci_regions__both_extracted(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
        ).all()

    def test__left__rows_cuts_out_bottom_row(self):

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(0, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
        ).all()

    def test__extract_two_regions_and_cut_bottom_row_from_each(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 5.0, 5.0, 5.0]])
        ).all()

    def test__extract_two_regions_and_cut_top_row_from_each(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        serial_frame = frame.serial_calibration_section_for_rows(rows=(1, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
        ).all()


class TestSerialCalibrationArrays:
    def test__left__different_ci_regions__extracts_all_columns(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_region = frame.serial_calibration_sub_arrays

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

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 4)])

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_region = frame.serial_calibration_sub_arrays

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

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_region = frame.serial_calibration_sub_arrays

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

    def test__right__ci_region_has_prescan_and_overscan(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        serial_region = frame.serial_calibration_sub_arrays

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

    def test__left__2_ci_regions__both_extracted(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        serial_region = frame.serial_calibration_sub_arrays

        assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
        assert (serial_region[1] == np.array([[0.0, 1.0, 4.0, 4.0, 4.0]])).all()


class TestParallelFrontEdgeArrays:
    def test__bottom___1_region__extracts_front_edges_correctly(self):

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edge = frame.parallel_front_edge_arrays(rows=(0, 1))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0]])).all()

        front_edge = frame.parallel_front_edge_arrays(rows=(1, 2))
        assert (front_edge[0] == np.array([[2.0, 2.0, 2.0]])).all()

        front_edge = frame.parallel_front_edge_arrays(rows=(2, 3))
        assert (front_edge[0] == np.array([[3.0, 3.0, 3.0]])).all()

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 5, 0, 3)])

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edge = frame.parallel_front_edge_arrays(rows=(0, 2))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

        front_edge = frame.parallel_front_edge_arrays(rows=(1, 4))
        assert (
            front_edge[0]
            == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

    def test__bottom__2_regions__extracts_rows_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edges = frame.parallel_front_edge_arrays(rows=(0, 1))
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(1, 2))
        assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
        assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(2, 3))
        assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(0, 3))
        assert (
            front_edges[0]
            == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        ).all()
        assert (
            front_edges[1]
            == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
        ).all()

    def test__top__does_all_the_above_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 5, 0, 3), (6, 9, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(0, 0), ci_pattern=ci_pattern)

        front_edges = frame.parallel_front_edge_arrays(rows=(0, 1))
        assert (front_edges[0] == np.array([[4.0, 4.0, 4.0]])).all()
        assert (front_edges[1] == np.array([[8.0, 8.0, 8.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(1, 2))
        assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(2, 3))
        assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
        assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

        front_edges = frame.parallel_front_edge_arrays(rows=(0, 3))
        assert (
            front_edges[0]
            == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()
        assert (
            front_edges[1]
            == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        # First fronrt edge arrays:
        #
        # [[1.0, 1.0, 1.0],
        #  [2.0, 2.0, 2.0],
        #  [3.0, 3.0, 3.0]])

        # Second front edge arrays:

        # [[5.0, 5.0, 5.0],
        #  [6.0, 6.0, 6.0],
        #  [7.0, 7.0, 7.0]]

        stacked_front_edges = frame.parallel_front_edge_stacked_array(rows=(0, 3))

        assert (
            stacked_front_edges
            == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        ).all()

        front_edge_line = frame.parallel_front_edge_line_binned_over_columns(
            rows=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    def test__no_rows_specified__uses_smallest_ci_pattern_rows(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edges = frame.parallel_front_edge_arrays()
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])).all()

        stacked_front_edges = frame.parallel_front_edge_stacked_array()

        assert (
            stacked_front_edges == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

        front_edge_line = frame.parallel_front_edge_line_binned_over_columns()

        assert (front_edge_line == np.array([3.0, 4.0])).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        front_edges = frame.parallel_front_edge_arrays(rows=(0, 3))

        assert (
            front_edges[0].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

        assert (
            front_edges[1].mask
            == np.array(
                [[False, False, False], [False, False, False], [True, False, False]]
            )
        ).all()

    def test__stacked_frame__include_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        # First fronrt edge arrays:
        #
        # [[1.0, 1.0, 1.0],
        #  [2.0, 2.0, 2.0],
        #  [3.0, 3.0, 3.0]])

        # Second front edge arrays:

        # [[5.0, 5.0, 5.0],
        #  [6.0, 6.0, 6.0],
        #  [7.0, 7.0, 7.0]]

        stacked_front_edges = frame.parallel_front_edge_stacked_array(rows=(0, 3))

        assert (
            stacked_front_edges
            == np.ma.array([[5.0, 3.0, 5.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
        ).all()
        assert (
            stacked_front_edges.mask
            == np.ma.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        front_edge_line = frame.parallel_front_edge_line_binned_over_columns(
            rows=(0, 3)
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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        # First fronrt edge arrays:
        #
        # [[1.0, 1.0, 1.0],
        #  [2.0, 2.0, 2.0],
        #  [3.0, 3.0, 3.0]])

        # Second front edge arrays:

        # [[5.0, 5.0, 5.0],
        #  [6.0, 6.0, 6.0],
        #  [7.0, 7.0, 7.0]]

        stacked_front_edges = frame.parallel_front_edge_stacked_array(rows=(0, 3))

        assert (
            stacked_front_edges.mask
            == np.ma.array(
                [[False, False, True], [False, False, False], [False, False, False]]
            )
        ).all()


class TestParallelTrailsArray:
    def test__bottom__1_region__extracts_trails_correctly(self):

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.parallel_trails_arrays(rows=(0, 1))

        assert (trails == np.array([[3.0, 3.0, 3.0]])).all()
        trails = frame.parallel_trails_arrays(rows=(1, 2))

        assert (trails == np.array([[4.0, 4.0, 4.0]])).all()
        trails = frame.parallel_trails_arrays(rows=(2, 3))
        assert (trails == np.array([[5.0, 5.0, 5.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(0, 2))
        assert (trails == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(1, 3))
        assert (trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(1, 4))
        assert (
            trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()

    def test__bottom__2_regions__extracts_rows_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.parallel_trails_arrays(rows=(0, 1))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(0, 2))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(1, 4))
        assert (
            trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()
        assert (
            trails[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
        ).all()

    def test__top__does_all_the_above_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(3, 5, 0, 3), (9, 10, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(0, 0), ci_pattern=ci_pattern)

        trails = frame.parallel_trails_arrays(rows=(0, 1))
        assert (trails[0] == np.array([[2.0, 2.0, 2.0]])).all()
        assert (trails[1] == np.array([[8.0, 8.0, 8.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(0, 2))
        assert (trails[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()
        assert (trails[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])).all()

        trails = frame.parallel_trails_arrays(rows=(1, 3))
        assert (trails[0] == np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        trails = frame.parallel_trails_arrays(rows=(0, 2))
        assert (
            trails[0].mask == np.array([[False, True, True], [False, False, False]])
        ).all()

        assert (
            trails[1].mask == np.array([[False, False, False], [True, False, False]])
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        # [3.0, 3.0, 3.0],
        #  [4.0, 4.0, 4.0]]

        # Array 2:

        # [[6.0, 6.0, 6.0],
        # [7.0, 7.0, 7.0]]

        stacked_trails = frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

        trails_line = frame.parallel_trails_line_binned_over_columns(rows=(0, 2))

        assert (trails_line == np.array([4.5, 5.5])).all()

    def test__same_as_above__include_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        # [3.0, 3.0, 3.0],
        #  [4.0, 4.0, 4.0]]

        # Array 2:

        # [[6.0, 6.0, 6.0],
        # [7.0, 7.0, 7.0]]

        stacked_trails = frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (stacked_trails == np.array([[4.5, 6.0, 6.0], [4.0, 5.5, 5.5]])).all()
        assert (
            stacked_trails.mask
            == np.array([[False, False, False], [False, False, False]])
        ).all()

        trails_line = frame.parallel_trails_line_binned_over_columns(rows=(0, 2))

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

        frame = ac.masked.ci_frame.manual(
            array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern
        )

        stacked_trails = frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (
            stacked_trails.mask
            == np.array([[False, False, True], [False, False, False]])
        ).all()

    def test__no_rows_specified__uses_smallest_parallel_trails_size(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(1, 4, 0, 3), (6, 8, 0, 3)]
        )

        arr = np.array(
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

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.parallel_trails_arrays()
        assert (trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()
        assert (trails[1] == np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])).all()

        stacked_trails = frame.parallel_trails_stacked_array()

        assert (stacked_trails == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails_line = frame.parallel_trails_line_binned_over_columns()

        assert (trails_line == np.array([6.0, 7.0])).all()


class TestSerialFrontEdgeArrays:
    def test__bottom___1_region__extracts_multiple_front_edges_correctly(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #       /| Front Edge

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edge = frame.serial_front_edge_arrays(columns=(0, 1))

        assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

        front_edge = frame.serial_front_edge_arrays(columns=(1, 2))

        assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

        front_edge = frame.serial_front_edge_arrays(columns=(2, 3))

        assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 5)])

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edge = frame.serial_front_edge_arrays(columns=(0, 2))

        assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        front_edge = frame.serial_front_edge_arrays(columns=(1, 4))

        assert (
            front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        ).all()

    def test__bottom__2_regions__extracts_columns_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                    /| FE 1        /\ FE 2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edges = frame.serial_front_edge_arrays(columns=(0, 1))

        assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(1, 2))

        assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(2, 3))

        assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(0, 3))

        assert (
            front_edges[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edges[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__pattern_right__does_all_the_above_correctly(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                               /| FE 1            /\ FE 2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        front_edges = frame.serial_front_edge_arrays(columns=(0, 1))

        assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(1, 2))

        assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(2, 3))

        assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edges = frame.serial_front_edge_arrays(columns=(0, 3))

        assert (
            front_edges[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edges[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                    /| FE 1        /\ FE 2

        mask = np.array(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, True, False, False, False, True, False, False, False],
                [False, False, False, False, False, False, False, True, False, False],
            ]
        )

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        front_edges = frame.serial_front_edge_arrays(columns=(0, 3))

        assert (
            (front_edges[0].mask)
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        assert (
            front_edges[1].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                      /| FE 1                /\ FE 2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        stacked_front_edges = frame.serial_front_edge_stacked_array(columns=(0, 3))

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

        front_edge_line = frame.serial_front_edge_line_binned_over_rows(columns=(0, 3))

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    def test__same_as_above__include_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                      /| FE 1                /\ FE 2

        mask = np.array(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, True, False, False, False],
                [False, False, False, False, False, False, False, True, False, False],
            ]
        )

        #                        /| FE 1                       /| FE 2

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        stacked_front_edges = frame.serial_front_edge_stacked_array(columns=(0, 3))

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
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        front_edge_line = frame.serial_front_edge_line_binned_over_rows(columns=(0, 3))

        assert (front_edge_line == np.array([3.0, 10.0 / 3.0, 5.0])).all()

        mask = np.array(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, True, True, False, False],
                [False, False, False, False, False, False, False, True, False, False],
            ]
        )

        #                        /| FE 1                       /| FE 2

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        stacked_front_edges = frame.serial_front_edge_stacked_array(columns=(0, 3))

        assert (
            stacked_front_edges.mask
            == np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            )
        ).all()

    def test__no_columns_specified_so_uses_smallest_charge_injection_region_column_size(
        self
    ):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 3), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                    /| FE 1        /\ FE 2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        front_edges = frame.serial_front_edge_arrays()

        assert (front_edges[0] == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        assert (front_edges[1] == np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])).all()

        stacked_front_edges = frame.serial_front_edge_stacked_array()

        assert (
            stacked_front_edges == np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
        ).all()

        front_edge_line = frame.serial_front_edge_line_binned_over_rows()

        assert (front_edge_line == np.array([3.0, 4.0])).all()


class TestSerialTrailsArrays:
    def test__bottom___1_region__extracts_multiple_trails_correctly(self):
        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                                    /| Trails Begin

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.serial_trails_arrays(columns=(0, 1))

        assert (trails == np.array([[4.0], [4.0], [4.0]])).all()

        trails = frame.serial_trails_arrays(columns=(1, 2))

        assert (trails == np.array([[5.0], [5.0], [5.0]])).all()

        trails = frame.serial_trails_arrays(columns=(2, 3))

        assert (trails == np.array([[6.0], [6.0], [6.0]])).all()

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.serial_trails_arrays(columns=(0, 2))

        assert (trails == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

        trails = frame.serial_trails_arrays(columns=(1, 4))

        assert (
            trails == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__bottom__2_regions__extracts_columns_correctly(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )

        #                                   /| Trails1           /\ Trails2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.serial_trails_arrays(columns=(0, 1))

        assert (trails[0] == np.array([[4.0], [4.0], [4.0]])).all()
        assert (trails[1] == np.array([[8.0], [8.0], [8.0]])).all()

        trails = frame.serial_trails_arrays(columns=(1, 2))

        assert (trails[0] == np.array([[5.0], [5.0], [5.0]])).all()
        assert (trails[1] == np.array([[9.0], [9.0], [9.0]])).all()

        trails = frame.serial_trails_arrays(columns=(2, 3))

        assert (trails[0] == np.array([[6.0], [6.0], [6.0]])).all()
        assert (trails[1] == np.array([[10.0], [10.0], [10.0]])).all()

        trails = frame.serial_trails_arrays(columns=(0, 3))

        assert (
            trails[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        ).all()

        assert (
            trails[1]
            == np.array([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
        ).all()

    def test__pattern_right__does_all_the_above_correctly(self):
        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 3, 6), (0, 3, 8, 11)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )

        #               Trails1   /|                Trails2 /\

        frame = ac.ci_frame.manual(array=arr, corner=(1, 1), ci_pattern=ci_pattern)

        trails = frame.serial_trails_arrays(columns=(0, 1))

        assert (trails[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (trails[1] == np.array([[7.0], [7.0], [7.0]])).all()

        trails = frame.serial_trails_arrays(columns=(1, 2))

        assert (trails[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (trails[1] == np.array([[6.0], [6.0], [6.0]])).all()

        trails = frame.serial_trails_arrays(columns=(2, 3))

        assert (trails[0] == np.array([[0.0], [0.0], [0.0]])).all()
        assert (trails[1] == np.array([[5.0], [5.0], [5.0]])).all()

        trails = frame.serial_trails_arrays(columns=(0, 3))

        assert (
            trails[0] == np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        ).all()

        assert (
            trails[1] == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        trails = frame.serial_trails_arrays(columns=(0, 3))

        assert (
            trails[0].mask
            == np.array(
                [[False, False, False], [True, False, True], [False, False, False]]
            )
        ).all()

        assert (
            trails[1].mask
            == np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            )
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )

        #                                   /| Trails1           /\ Trails2

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = frame.serial_trails_stacked_array(columns=(0, 3))

        assert (
            stacked_trails
            == np.array([[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 7.0, 8.0]])
        ).all()

        trails_line = frame.serial_trails_line_binned_over_rows(columns=(0, 3))

        assert (trails_line == np.array([6.0, 7.0, 8.0])).all()

    def test__same_as_above__include_masking(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = np.array(
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

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = frame.serial_trails_stacked_array(columns=(0, 3))

        assert (
            stacked_trails
            == np.array([[6.0, 7.0, 8.0], [8.0, 7.0, 10.0], [6.0, 5.0, 8.0]])
        ).all()
        assert (
            stacked_trails.mask
            == np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        trails_line = frame.serial_trails_line_binned_over_rows(columns=(0, 3))

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

        frame = ac.masked.ci_frame.manual(array=arr, mask=mask, corner=(1, 0), ci_pattern=ci_pattern)

        stacked_trails = frame.serial_trails_stacked_array(columns=(0, 3))

        assert (
            stacked_trails.mask
            == np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            )
        ).all()

    def test__no_columns_specified_so_uses_full_serial_overscan(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 4, 7)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ]
        )

        #                                      /| Trails1           /\ Trails2

        ci_frame = ac.ci_frame.manual(
            array=arr,
            ci_pattern=ci_pattern,
            serial_overscan=ac.Region((0, 1, 0, 4)),
            serial_prescan=ac.Region((0, 1, 0, 1)),
            parallel_overscan=ac.Region((0, 1, 0, 1)),
            corner=(0, 0),
        )

        trails = ci_frame.serial_trails_arrays()

        assert (
            trails[0]
            == np.array(
                [[4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0]]
            )
        ).all()

        assert (
            trails[1]
            == np.array(
                [[7.0, 8.0, 9.0, 10.0], [7.0, 8.0, 9.0, 10.0], [7.0, 8.0, 9.0, 10.0]]
            )
        ).all()

        stacked_trails = ci_frame.serial_trails_stacked_array()

        assert (
            stacked_trails
            == np.array(
                [[5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5]]
            )
        ).all()

        trails_line = ci_frame.serial_trails_line_binned_over_rows()

        assert (trails_line == np.array([5.5, 6.5, 7.5, 8.5])).all()


class TestParallelSerialCalibrationSection:
    def test__extracts_everything(self):

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        frame.frame_geometry.serial_prescan = ac.Region(region=(0, 4, 0, 1))

        extracted_array = frame.parallel_serial_calibration_section(arr)

        assert (extracted_array == arr).all()

        ci_pattern = ac.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )

        frame = ac.ci_frame.manual(array=arr, corner=(1, 0), ci_pattern=ci_pattern)

        frame.frame_geometry.serial_prescan = ac.Region(region=(0, 4, 0, 2))

        extracted_array = frame.parallel_serial_calibration_section(arr)

        assert (extracted_array == arr).all()


class TestSmallestPArallelTrailsRowsToEdge:
    def test__x1_ci_region__bottom_frame_geometry(self):

        ci_pattern = ac.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        frame = ac.ci_frame.manual(
            array=np.ones((5, 5)), corner=(1, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 2

        frame = ac.ci_frame.manual(
            array=np.ones((7, 5)), corner=(1, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 4

    def test__x2_ci_region__bottom_frame_geometry(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(0, 3, 0, 3), (5, 7, 0, 3)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((10, 5)), corner=(1, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 2

        frame = ac.ci_frame.manual(
            array=np.ones((8, 5)), corner=(1, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 1

    def test__x2_ci_region__top_frame_geometry(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(1, 4, 0, 3), (5, 7, 0, 3)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones((10, 5)), corner=(0, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 1

        ci_pattern = ac.CIPatternUniform(
            normalization=10.0, regions=[(8, 12, 0, 3), (14, 16, 0, 3)]
        )

        frame = ac.ci_frame.manual(
            array=np.ones(10, 5), corner=(0, 0), ci_pattern=ci_pattern
        )

        assert frame.smallest_parallel_trails_rows_to_frame_edge == 2


class TestSerialTrailsColumns:
    def test__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
        self
    ):

        ci_frame = ac.FrameGeometry(
            serial_overscan=ac.Region((0, 1, 0, 10)),
            serial_prescan=ac.Region((0, 1, 0, 1)),
            parallel_overscan=ac.Region((0, 1, 0, 1)),
            corner=(0, 0),
        )

        assert ci_frame.serial_trails_columns == 10

        ci_frame = ac.FrameGeometry(
            serial_overscan=ac.Region((0, 1, 0, 50)),
            serial_prescan=ac.Region((0, 1, 0, 1)),
            parallel_overscan=ac.Region((0, 1, 0, 1)),
            corner=(0, 0),
        )

        assert ci_frame.serial_trails_columns == 50


class TestParallelTrailsSizeToFrameEdge:
    def test__top_left__parallel_trail_size_to_image_edge(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[ac.Region(region=(0, 3, 0, 3))]
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((5, 100)), ci_pattern=ci_pattern, corner=(0, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 0

        ci_frame = ac.ci_frame.manual(
            array=np.ones((7, 100)), ci_pattern=ci_pattern, corner=(0, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 0

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0,
            regions=[
                ac.Region(region=(5, 6, 0, 3)),
                ac.Region(region=(11, 12, 0, 3)),
                ac.Region(region=(17, 18, 0, 3)),
            ],
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((20, 100)), ci_pattern=ci_pattern, corner=(0, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 5

    def test__top_right__parallel_trail_size_to_image_edge(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[ac.Region(region=(0, 3, 0, 3))]
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((5, 100)), ci_pattern=ci_pattern, corner=(0, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 0

        ci_frame = ac.ci_frame.manual(
            array=np.ones((7, 100)), ci_pattern=ci_pattern, corner=(0, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 0

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0,
            regions=[
                ac.Region(region=(5, 6, 0, 3)),
                ac.Region(region=(11, 12, 0, 3)),
                ac.Region(region=(17, 18, 0, 3)),
            ],
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((20, 100)), ci_pattern=ci_pattern, corner=(0, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 5

    def test__bottom_left__parallel_trail_size_to_image_edge(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[ac.Region(region=(0, 3, 0, 3))]
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((5, 100)), ci_pattern=ci_pattern, corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 2

        ci_frame = ac.ci_frame.manual(
            array=np.ones((7, 100)), ci_pattern=ci_pattern, corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 4

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0,
            regions=[
                ac.Region(region=(0, 2, 0, 3)),
                ac.Region(region=(5, 8, 0, 3)),
                ac.Region(region=(11, 14, 0, 3)),
            ],
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((15, 100)), ci_pattern=ci_pattern, corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 1

        ci_frame = ac.ci_frame.manual(
            array=np.ones((20, 100)), ci_pattern=ci_pattern, corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 6

    def test__bottom_right__parallel_trail_size_to_image_edge(self):

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0, regions=[ac.Region(region=(0, 3, 0, 3))]
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((5, 100)), ci_pattern=ci_pattern, corner=(1, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 2

        ci_frame = ac.ci_frame.manual(
            array=np.ones((7, 100)), ci_pattern=ci_pattern, corner=(1, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 4

        ci_pattern = ac.CIPatternUniform(
            normalization=1.0,
            regions=[
                ac.Region(region=(0, 2, 0, 3)),
                ac.Region(region=(5, 8, 0, 3)),
                ac.Region(region=(11, 14, 0, 3)),
            ],
        )

        ci_frame = ac.ci_frame.manual(
            array=np.ones((15, 100)), ci_pattern=ci_pattern, corner=(1, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 1

        ci_frame = ac.ci_frame.manual(
            array=np.ones((20, 100)), ci_pattern=ci_pattern, corner=(1, 1)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 6
