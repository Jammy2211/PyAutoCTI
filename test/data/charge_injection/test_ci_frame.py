import numpy as np

from autocti.data import mask as msk
from autocti.data.charge_injection import ci_frame, ci_pattern


class MockCIGeometry(object):

    def __init__(self, serial_prescan=(0, 1, 0, 1), serial_overscan=(0, 1, 0, 1)):
        super(MockCIGeometry, self).__init__()
        self.serial_prescan = ci_frame.Region(serial_prescan)
        self.serial_overscan = ci_frame.Region(serial_overscan)


class MockGeometry(object):

    def __init__(self):
        super(MockGeometry, self).__init__()


class MockRegion(tuple):

    def __new__(cls, region):
        region = super(MockRegion, cls).__new__(cls, region)

        region.y0 = region[0]
        region.y1 = region[1]
        region.x0 = region[2]
        region.x1 = region[3]

        return region


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
        image = np.array([[1.0, 2.0, 3.0],
                          [6.0, 6.0, 6.0],
                          [9.0, 9.0, 9.0]])

        binned_array = ci_frame.bin_array_across_serial(image)

        assert (binned_array == np.array([2.0, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_including_mask(self):
        image = np.array([[1.0, 2.0, 3.0],
                          [6.0, 6.0, 6.0],
                          [9.0, 9.0, 9.0]])

        mask = np.ma.array([[False, False, True],
                            [False, False, False],
                            [False, False, False]])

        binned_array = ci_frame.bin_array_across_serial(image, mask)

        assert (binned_array == np.array([1.5, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_an_entire_row_is_masked(self):
        image = np.array([[1.0, 2.0, 3.0],
                          [6.0, 6.0, 6.0],
                          [9.0, 9.0, 9.0]])

        mask = np.ma.array([[False, False, True],
                            [True, True, True],
                            [False, False, False]])

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
        image = np.array([[1.0, 6.0, 9.0],
                          [2.0, 6.0, 9.0],
                          [3.0, 6.0, 9.0]])

        binned_array = ci_frame.bin_array_across_parallel(image)

        assert (binned_array == np.array([2.0, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_with_mask(self):
        image = np.array([[1.0, 6.0, 9.0],
                          [2.0, 6.0, 9.0],
                          [3.0, 6.0, 9.0]])

        mask = np.ma.array([[False, False, False],
                            [False, False, False],
                            [True, False, False]])

        binned_array = ci_frame.bin_array_across_parallel(image, mask)

        assert (binned_array == np.array([1.5, 6.0, 9.0])).all()

    def test__3x3_array__same_as_above_but_with_entire_column_masked(self):
        image = np.array([[1.0, 6.0, 9.0],
                          [2.0, 6.0, 9.0],
                          [3.0, 6.0, 9.0]])

        mask = np.ma.array([[False, True, False],
                            [False, True, False],
                            [True, True, False]])

        binned_array = ci_frame.bin_array_across_parallel(image, mask)

        assert (binned_array == np.array([1.5, np.inf, 9.0])).all()


class TestChInj(object):

    def test__init__input_ci_data_grid_single_value__all_attributes_correct_including_ci_data_inheritance(self):
        pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

        frame = ci_frame.CIFrame.from_single_value(value=5.0, shape=(3, 3), frame_geometry=MockCIGeometry(),
                                                   ci_pattern=pattern)

        assert (frame == 5.0 * np.ones((3, 3))).all()
        assert frame.shape == (3, 3)
        assert type(frame.frame_geometry) == MockCIGeometry
        assert type(frame.ci_pattern) == ci_pattern.CIPattern


class TestCIFrame(object):
    class TestAllFunctionsReturnClassType:

        def test__frame_in_frame_out(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.ci_regions_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.parallel_non_ci_regions_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.parallel_edges_and_trails_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.serial_all_trails_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.serial_overscan_non_trails_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.serial_edges_and_trails_frame_from_frame()

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.serial_calibration_section_for_column_and_rows(column=0, rows=(0, 3))

            assert type(new_frame) == ci_frame.CIFrame
            assert new_frame.frame_geometry == frame.frame_geometry
            assert new_frame.ci_pattern == frame.ci_pattern

            new_frame = frame.parallel_front_edge_arrays_from_frame(rows=(0, 1))

            assert type(new_frame[0]) == ci_frame.CIFrame
            assert new_frame[0].frame_geometry == frame.frame_geometry
            assert new_frame[0].ci_pattern == frame.ci_pattern

            new_frame = frame.parallel_trails_arrays_from_frame(rows=(0, 1))

            assert type(new_frame[0]) == ci_frame.CIFrame
            assert new_frame[0].frame_geometry == frame.frame_geometry
            assert new_frame[0].ci_pattern == frame.ci_pattern

            new_frame = frame.serial_front_edge_arrays_from_frame(columns=(0, 1))

            assert type(new_frame[0]) == ci_frame.CIFrame
            assert new_frame[0].frame_geometry == frame.frame_geometry
            assert new_frame[0].ci_pattern == frame.ci_pattern

            new_frame = frame.serial_trails_arrays_from_frame(columns=(0, 1))

            assert type(new_frame[0]) == ci_frame.CIFrame
            assert new_frame[0].frame_geometry == frame.frame_geometry
            assert new_frame[0].ci_pattern == frame.ci_pattern

    class TestCiRegionArrayFromFrame:

        def test__1_ci_region__extracted_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 1.0, 2.0],
                                           [3.0, 4.0, 5.0],
                                           [6.0, 7.0, 8.0],
                                           [0.0, 0.0, 0.0]])).all()

        def test__2_ci_regions__extracted_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 1, 2), (2, 3, 1, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 7.0, 8.0],
                                           [0.0, 0.0, 0.0]])).all()

    class TestCIParallelNonRegionArrayFromFrame:

        def test__1_ci_region__pre_scan_and_overscan_in_corner__extracts_everything_outside_region_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(
                frame_geometry=MockCIGeometry(serial_prescan=(0, 1, 0, 1), serial_overscan=(0, 1, 0, 1)),
                ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                    [3.0, 4.0, 5.0],
                                                    [6.0, 7.0, 8.0],
                                                    [9.0, 10.0, 11.0]]))

            new_frame = frame.parallel_non_ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [9.0, 10.0, 11.0]])).all()

        def test__2_ci_regions__pre_scan_and_overscan_in_corner__extracts_everything_outside_region_correctly(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)])

            frame = ci_frame.CIFrame(
                frame_geometry=MockCIGeometry(serial_prescan=(0, 1, 0, 1), serial_overscan=(0, 1, 0, 1)),
                ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                    [3.0, 4.0, 5.0],
                                                    [6.0, 7.0, 8.0],
                                                    [9.0, 10.0, 11.0],
                                                    [12.0, 13.0, 14.0]]))

            new_frame = frame.parallel_non_ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [3.0, 4.0, 5.0],
                                           [6.0, 7.0, 8.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0]])).all()

        def test__2_ci_regions__serial_prescan_overlaps_an_extraction__extraction_goes_to_0(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=MockCIGeometry(serial_prescan=(1, 2, 0, 2),
                                                                   serial_overscan=(0, 1, 0, 1)),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0]]))

            new_frame = frame.parallel_non_ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 5.0],
                                           [6.0, 7.0, 8.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0]])).all()

        def test__2_ci_regions__serial_overscan_overlaps_an_extraction__extraction_goes_to_0(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=MockCIGeometry(serial_prescan=(0, 1, 0, 1),
                                                                   serial_overscan=(1, 2, 1, 3)),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0]]))

            new_frame = frame.parallel_non_ci_regions_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [3.0, 0.0, 0.0],
                                           [6.0, 7.0, 8.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0]])).all()

    class TestParallelEdgesAndTrailsArrayFromFrame:

        def test__front_edge_only__1_row__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(front_edge_rows=(0, 1))

            assert (new_frame == np.array([[0.0, 1.0, 2.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])).all()

        def test__front_edge_only__2_rows__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(front_edge_rows=(0, 2))

            assert (new_frame == np.array([[0.0, 1.0, 2.0],
                                           [3.0, 4.0, 5.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]])).all()

        def test__trails_only__1_row__new_frame_is_just_that_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(trails_rows=(0, 1))

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0]])).all()

        def test__trails_only__2_rows__new_frame_is_the_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0],
                                                                         [15.0, 16.0, 17.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(trails_rows=(0, 2))

            assert (new_frame == np.array([[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0],
                                           [15.0, 16.0, 17.0]])).all()

        def test__front_edge_and_trails__2_rows_of_each__new_frame_is_edge_and_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0],
                                                                         [15.0, 16.0, 17.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(front_edge_rows=(0, 2), trails_rows=(0, 2))

            assert (new_frame == np.array([[0.0, 1.0, 2.0],
                                           [3.0, 4.0, 5.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [12.0, 13.0, 14.0],
                                           [15.0, 16.0, 17.0]])).all()

        def test__front_edge_and_trails__2_regions__1_row_of_each__new_frame_is_edge_and_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 3), (3, 4, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0],
                                                                         [15.0, 16.0, 17.0]]))

            new_frame = frame.parallel_edges_and_trails_frame_from_frame(front_edge_rows=(0, 1), trails_rows=(0, 1))

            assert (new_frame == np.array([[0.0, 1.0, 2.0],
                                           [3.0, 4.0, 5.0],
                                           [0.0, 0.0, 0.0],
                                           [9.0, 10.0, 11.0],
                                           [12.0, 13.0, 14.0],
                                           [0.0, 0.0, 0.0]])).all()

    class TesParallelCalibrationSectionFromFrame:

        def test__geometry_left__columns_0_to_1__extracts_1_column_left_hand_side_of_array(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

            image = np.array([[0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            extracted_side = frame.parallel_calibration_section_for_columns(columns=(0, 1))

            assert (extracted_side == np.array([[0.0],
                                                [0.0],
                                                [0.0],
                                                [0.0],
                                                [0.0]])).all()

        def test__geometry_bottom__columns_1_to_3__extracts_2_columns_middle_and_right_of_array(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

            image = np.array([[0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            extracted_side = frame.parallel_calibration_section_for_columns(columns=(1, 3))

            assert (extracted_side == np.array([[1.0, 2.0],
                                                [1.0, 2.0],
                                                [1.0, 2.0],
                                                [1.0, 2.0],
                                                [1.0, 2.0]])).all()

        def test__geometry_right__columns_1_to_3__extracts_2_columns_middle_and_left_of_array(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

            image = np.array([[0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0],
                              [0.0, 1.0, 2.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            extracted_side = frame.parallel_calibration_section_for_columns(columns=(1, 3))

            assert (extracted_side == np.array([[0.0, 1.0],
                                                [0.0, 1.0],
                                                [0.0, 1.0],
                                                [0.0, 1.0],
                                                [0.0, 1.0]])).all()

    class TestKeepSerialEdgesAndTrailsArrayFromFrame:

        def test__front_edge_only__1_column__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(front_edge_columns=(0, 1))

            assert (new_frame == np.array([[0.0, 0.0, 0.0, 0.0],
                                           [4.0, 0.0, 0.0, 0.0],
                                           [8.0, 0.0, 0.0, 0.0]])).all()

        def test__front_edge_only__2_columns__new_frame_is_just_that_edge(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 3)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(front_edge_columns=(0, 2))

            assert (new_frame == np.array([[0.0, 1.0, 0.0, 0.0],
                                           [4.0, 5.0, 0.0, 0.0],
                                           [8.0, 9.0, 0.0, 0.0]])).all()

        def test__trails_only__1_column__new_frame_is_just_that_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(trails_columns=(0, 1))

            assert (new_frame == np.array([[0.0, 0.0, 2.0, 0.0],
                                           [0.0, 0.0, 6.0, 0.0],
                                           [0.0, 0.0, 10.0, 0.0]])).all()

        def test__trails_only__2_columns__new_frame_is_the_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(trails_columns=(0, 2))

            assert (new_frame == np.array([[0.0, 0.0, 2.0, 3.0],
                                           [0.0, 0.0, 6.0, 7.0],
                                           [0.0, 0.0, 10.0, 11.0]])).all()

        def test__front_edge_and_trails__2_columns_of_each__new_frame_is_edge_and_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 2)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 1.1, 2.0, 3.0],
                                                                         [4.0, 5.0, 1.1, 6.0, 7.0],
                                                                         [8.0, 9.0, 1.1, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(front_edge_columns=(0, 1), trails_columns=(0, 2))

            assert (new_frame == np.array([[0.0, 0.0, 1.1, 2.0, 0.0],
                                           [4.0, 0.0, 1.1, 6.0, 0.0],
                                           [8.0, 0.0, 1.1, 10.0, 0.0]])).all()

        def test__front_edge_and_trails__2_regions_1_column_of_each__new_frame_is_edge_and_trail(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 3, 0, 1), (0, 3, 3, 4)])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(),
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 1.1, 2.0, 3.0],
                                                                         [4.0, 5.0, 1.1, 6.0, 7.0],
                                                                         [8.0, 9.0, 1.1, 10.0, 11.0]]))

            new_frame = frame.serial_edges_and_trails_frame_from_frame(front_edge_columns=(0, 1), trails_columns=(0, 1))

            assert (new_frame == np.array([[0.0, 1.0, 0.0, 2.0, 3.0],
                                           [4.0, 5.0, 0.0, 6.0, 7.0],
                                           [8.0, 9.0, 0.0, 10.0, 11.0]])).all()

    class TestSerialAllTrailsArrayFromFrame:

        def test__left_quadrant__1_ci_region__1_serial_trail__extracts_all_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 2)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 3))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0],
                                                                         [3.0, 4.0, 5.0],
                                                                         [6.0, 7.0, 8.0],
                                                                         [9.0, 10.0, 11.0]]))

            new_frame = frame.serial_all_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 2.0],
                                           [0.0, 0.0, 5.0],
                                           [0.0, 0.0, 8.0],
                                           [0.0, 0.0, 11.0]])).all()

        def test__left_quadrant__1_ci_region__2_serial_trail__extracts_all_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 4, 0, 2)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 4))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 0.5],
                                                                         [3.0, 4.0, 5.0, 0.5],
                                                                         [6.0, 7.0, 8.0, 0.5],
                                                                         [9.0, 10.0, 11.0, 0.5]]))

            new_frame = frame.serial_all_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 2.0, 0.5],
                                           [0.0, 0.0, 5.0, 0.5],
                                           [0.0, 0.0, 8.0, 0.5],
                                           [0.0, 0.0, 11.0, 0.5]])).all()

        def test__left_quadrant__2_ci_regions__2_serial_trail__extracts_all_trails(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 0, 2), (2, 3, 0, 2)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 2, 4))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 0.5],
                                                                         [3.0, 4.0, 5.0, 0.5],
                                                                         [6.0, 7.0, 8.0, 0.5],
                                                                         [9.0, 10.0, 11.0, 0.5]]))

            new_frame = frame.serial_all_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 2.0, 0.5],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 8.0, 0.5],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

        def test__same_as_above_but_right_quadrant__flips_trails_side(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(0, 1, 2, 4), (2, 3, 2, 4)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            frame_geometry.serial_overscan = ci_frame.Region((0, 4, 0, 2))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 0.5],
                                                                         [3.0, 4.0, 5.0, 0.5],
                                                                         [6.0, 7.0, 8.0, 0.5],
                                                                         [9.0, 10.0, 11.0, 0.5]]))

            new_frame = frame.serial_all_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 1.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [6.0, 7.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

    class TestSerialOverScanNonTrailsFromFrame:

        def test__left_quadrant__1_ci_region__serial_trails_go_over_2_right_hand_columns__2_pixels_above_kept(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 3, 1, 2)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 3, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 3, 2, 4))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_overscan_non_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 2.0, 3.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

        def test__left_quadrant__1_ci_region__serial_trails_go_over_1_right_hand_column__1_pixel_above_kept(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 3, 1, 3)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 3, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 3, 3, 4))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0]]))

            new_frame = frame.serial_overscan_non_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0, 3.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0]])).all()

        def test__left_quadrant__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            frame_geometry.serial_prescan = ci_frame.Region((0, 5, 0, 1))
            frame_geometry.serial_overscan = ci_frame.Region((0, 5, 3, 4))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.0, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0, 15.0],
                                                                         [16.0, 17.0, 18.0, 19.0]]))

            new_frame = frame.serial_overscan_non_trails_frame_from_frame()

            assert (new_frame == np.array([[0.0, 0.0, 0.0, 3.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 11.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 19.0]])).all()

        def test__right_quadrant__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
                self):
            pattern = ci_pattern.CIPattern(normalization=10.0, regions=[(1, 2, 1, 3), (3, 4, 1, 3)])

            frame_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            frame_geometry.serial_prescan = ci_frame.Region((0, 5, 3, 4))
            frame_geometry.serial_overscan = ci_frame.Region((0, 5, 0, 1))

            frame = ci_frame.CIFrame(frame_geometry=frame_geometry,
                                     ci_pattern=pattern, array=np.array([[0.5, 1.0, 2.0, 3.0],
                                                                         [4.0, 5.0, 6.0, 7.0],
                                                                         [8.0, 9.0, 10.0, 11.0],
                                                                         [12.0, 13.0, 14.0, 15.0],
                                                                         [16.0, 17.0, 18.0, 19.0]]))

            new_frame = frame.serial_overscan_non_trails_frame_from_frame()

            assert (new_frame == np.array([[0.5, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [8.0, 0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0, 0.0],
                                           [16.0, 0.0, 0.0, 0.0]])).all()

    class TestSerialCalibrationArrayFromFrame:

        def test__geometry_left__ci_region_across_all_image__column_0__extracts_all_columns(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=0, rows=(0, 3))

            assert (serial_frame[0] == np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                                                 [0.0, 1.0, 2.0, 3.0, 4.0],
                                                 [0.0, 1.0, 2.0, 3.0, 4.0]])).all()

        def test__geometry_left__2_ci_regions__both_extracted(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 2.0, 2.0],
                              [0.0, 1.0, 3.0, 3.0, 3.0],
                              [0.0, 1.0, 4.0, 4.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=1, rows=(0, 1))

            assert (serial_frame == np.array([[2.0, 2.0, 2.0],
                                              [4.0, 4.0, 4.0]])).all()

        def test__geometry_right__2_ci_regions__both_extracted(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 2.0, 2.0],
                              [0.0, 1.0, 3.0, 3.0, 3.0],
                              [0.0, 1.0, 4.0, 4.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=1, rows=(0, 1))

            assert (serial_frame == np.array([[0.0, 1.0, 2.0],
                                              [0.0, 1.0, 4.0]])).all()

        def test__geometry_left__rows_cuts_out_bottom_row(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=0, rows=(0, 2))

            assert (serial_frame == np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                                              [0.0, 1.0, 2.0, 3.0, 4.0]])).all()

        def test__extract_two_regions_and_cut_bottom_row_from_each(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 2.0, 2.0],
                              [0.0, 1.0, 3.0, 3.0, 3.0],
                              [0.0, 1.0, 4.0, 4.0, 4.0],
                              [0.0, 1.0, 5.0, 5.0, 5.0],
                              [0.0, 1.0, 6.0, 6.0, 6.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=1, rows=(0, 1))

            assert (serial_frame == np.array([[0.0, 1.0, 2.0],
                                              [0.0, 1.0, 5.0]])).all()

        def test__extract_two_regions_and_cut_top_row_from_each(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 2, 1, 4), (3, 5, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 2.0, 2.0],
                              [0.0, 1.0, 3.0, 3.0, 3.0],
                              [0.0, 1.0, 4.0, 4.0, 4.0],
                              [0.0, 1.0, 5.0, 5.0, 5.0],
                              [0.0, 1.0, 6.0, 6.0, 6.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            serial_frame = frame.serial_calibration_section_for_column_and_rows(column=1, rows=(1, 2))

            assert (serial_frame == np.array([[0.0, 1.0, 3.0],
                                              [0.0, 1.0, 6.0]])).all()

    class TestSerialCalibrationSubArrays:

        def test__geometry_left__ci_region_across_all_image__column_0__extracts_all_columns(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=0)

            assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                                                  [0.0, 1.0, 2.0, 3.0, 4.0],
                                                  [0.0, 1.0, 2.0, 3.0, 4.0]])).all()

        def test__geometry_left__ci_region_misses_serial_overscan__column_0__extracts_all_columns(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=0)

            assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                                                  [0.0, 1.0, 2.0, 3.0, 4.0],
                                                  [0.0, 1.0, 2.0, 3.0, 4.0]])).all()

        def test__geometry_left__ci_region_also_has_prescan__extracts_all_but_1_column(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=0)

            assert (serial_region[0] == np.array([[1.0, 2.0, 3.0, 4.0],
                                                  [1.0, 2.0, 3.0, 4.0],
                                                  [1.0, 2.0, 3.0, 4.0]])).all()

        def test__geometry_left__same_as_above_but_column_2(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=1)

            assert (serial_region[0] == np.array([[2.0, 3.0, 4.0],
                                                  [2.0, 3.0, 4.0],
                                                  [2.0, 3.0, 4.0]])).all()

        def test__geometry_right__ci_region_has_prescan_and_overscan(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=0)

            assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 3.0],
                                                  [0.0, 1.0, 2.0, 3.0],
                                                  [0.0, 1.0, 2.0, 3.0]])).all()

        def test__geometry_right__also_include_column(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=1)

            assert (serial_region[0] == np.array([[0.0, 1.0, 2.0],
                                                  [0.0, 1.0, 2.0],
                                                  [0.0, 1.0, 2.0]])).all()

        def test__geometry_left__2_ci_regions__both_extracted(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 2.0, 2.0],
                              [0.0, 1.0, 3.0, 3.0, 3.0],
                              [0.0, 1.0, 4.0, 4.0, 4.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            serial_region = frame.serial_calibration_sub_arrays_from_frame(column=1)

            assert (serial_region[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (serial_region[1] == np.array([[4.0, 4.0, 4.0]])).all()

    class TestExtractParallelFrontEdges:

        def test__pattern_bottom___extracts_1_front_edge_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],  # <- Front edge according to region and this frame_geometry
                              [2.0, 2.0, 2.0],  # <- Next front edge row.
                              [3.0, 3.0, 3.0],
                              [4.0, 4.0, 4.0],
                              [5.0, 5.0, 5.0],
                              [6.0, 6.0, 6.0],
                              [7.0, 7.0, 7.0],
                              [8.0, 8.0, 8.0],
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edge = frame.parallel_front_edge_arrays_from_frame(rows=(0, 1))

            assert (front_edge == np.array([[1.0, 1.0, 1.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(rows=(1, 2))

            assert (front_edge == np.array([[2.0, 2.0, 2.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(rows=(2, 3))

            assert (front_edge == np.array([[3.0, 3.0, 3.0]])).all()

        def test__pattern_bottom___extracts_multiple_front_edges_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 5, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],  # <- Front edge according to region and this frame_geometry
                              [2.0, 2.0, 2.0],  # <- Next front edge row.
                              [3.0, 3.0, 3.0],
                              [4.0, 4.0, 4.0],
                              [5.0, 5.0, 5.0],
                              [6.0, 6.0, 6.0],
                              [7.0, 7.0, 7.0],
                              [8.0, 8.0, 8.0],
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edge = frame.parallel_front_edge_arrays_from_frame(rows=(0, 2))

            assert (front_edge == np.array([[1.0, 1.0, 1.0],
                                            [2.0, 2.0, 2.0]])).all()

            front_edge = frame.parallel_front_edge_arrays_from_frame(rows=(1, 4))

            assert (front_edge == np.array([[2.0, 2.0, 2.0],
                                            [3.0, 3.0, 3.0],
                                            [4.0, 4.0, 4.0]])).all()

        def test__pattern_bottom__2_regions__extracts_rows_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3), (5, 8, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],  # <- 1st Front edge according to region and this frame_geometry
                              [2.0, 2.0, 2.0],  # <- Next front edge row.
                              [3.0, 3.0, 3.0],
                              [4.0, 4.0, 4.0],
                              [5.0, 5.0, 5.0],  # <- 2nd Front edge according to region and this frame_geometry
                              [6.0, 6.0, 6.0],  # <- Next front edge row.
                              [7.0, 7.0, 7.0],
                              [8.0, 8.0, 8.0],
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(0, 1))

            assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
            assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(1, 2))

            assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(2, 3))

            assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(0, 3))
            assert (front_edges[0] == np.array([[1.0, 1.0, 1.0],
                                                [2.0, 2.0, 2.0],
                                                [3.0, 3.0, 3.0]])).all()
            assert (front_edges[1] == np.array([[5.0, 5.0, 5.0],
                                                [6.0, 6.0, 6.0],
                                                [7.0, 7.0, 7.0]])).all()

        def test__pattern_top__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 5, 0, 3), (6, 9, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0],  # <- Next front edge row.
                              [4.0, 4.0, 4.0],  # <- 1st Front edge according to region and this frame_geometry
                              [5.0, 5.0, 5.0],
                              [6.0, 6.0, 6.0],
                              [7.0, 7.0, 7.0],  # <- Next front edge row.
                              [8.0, 8.0, 8.0],  # <- 2nd Front edge according to region and this frame_geometry
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.top_left(), ci_pattern=pattern,
                                     array=image)

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(0, 1))

            assert (front_edges[0] == np.array([[4.0, 4.0, 4.0]])).all()
            assert (front_edges[1] == np.array([[8.0, 8.0, 8.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(1, 2))

            assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(2, 3))

            assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

            front_edges = frame.parallel_front_edge_arrays_from_frame(rows=(0, 3))
            assert (front_edges[0] == np.array([[2.0, 2.0, 2.0],
                                                [3.0, 3.0, 3.0],
                                                [4.0, 4.0, 4.0]])).all()
            assert (front_edges[1] == np.array([[6.0, 6.0, 6.0],
                                                [7.0, 7.0, 7.0],
                                                [8.0, 8.0, 8.0]])).all()

    class TestExtractParallelTrails:

        def test__pattern_bottom__extracts_1_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0],
                              # <- Trails form here onwards according to region and this frame_geometry
                              [4.0, 4.0, 4.0],  # <- Next trail.
                              [5.0, 5.0, 5.0],
                              [6.0, 6.0, 6.0],
                              [7.0, 7.0, 7.0],
                              [8.0, 8.0, 8.0],
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 1))

            assert (trails == np.array([[3.0, 3.0, 3.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(1, 2))

            assert (trails == np.array([[4.0, 4.0, 4.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(2, 3))

            assert (trails == np.array([[5.0, 5.0, 5.0]])).all()

        def test__pattern_bottom__extracts_multiple_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0],
                              [2.0, 2.0, 2.0],
                              [3.0, 3.0, 3.0],
                              # <- Trails form here onwards according to region and this frame_geometry
                              [4.0, 4.0, 4.0],  # <- Next trail.
                              [5.0, 5.0, 5.0],
                              [6.0, 6.0, 6.0],
                              [7.0, 7.0, 7.0],
                              [8.0, 8.0, 8.0],
                              [9.0, 9.0, 9.0]])
            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 2))

            assert (trails == np.array([[3.0, 3.0, 3.0],
                                        [4.0, 4.0, 4.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(1, 3))

            assert (trails == np.array([[4.0, 4.0, 4.0],
                                        [5.0, 5.0, 5.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(1, 4))

            assert (trails == np.array([[4.0, 4.0, 4.0],
                                        [5.0, 5.0, 5.0],
                                        [6.0, 6.0, 6.0]])).all()

        def test__pattern_bottom__2_regions__extracts_rows_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3), (4, 6, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
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
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 1))

            assert (trails[0] == np.array([[3.0, 3.0, 3.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 2))

            assert (trails[0] == np.array([[3.0, 3.0, 3.0],
                                           [4.0, 4.0, 4.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0],
                                           [7.0, 7.0, 7.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(1, 4))

            assert (trails[0] == np.array([[4.0, 4.0, 4.0],
                                           [5.0, 5.0, 5.0],
                                           [6.0, 6.0, 6.0]])).all()
            assert (trails[1] == np.array([[7.0, 7.0, 7.0],
                                           [8.0, 8.0, 8.0],
                                           [9.0, 9.0, 9.0]])).all()

        def test__pattern_top__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(3, 5, 0, 3), (9, 10, 0, 3)])

            image = np.array([[0.0, 0.0, 0.0],
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
                              [9.0, 9.0, 9.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.top_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 1))

            assert (trails[0] == np.array([[2.0, 2.0, 2.0]])).all()
            assert (trails[1] == np.array([[8.0, 8.0, 8.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(0, 2))

            assert (trails[0] == np.array([[1.0, 1.0, 1.0],
                                           [2.0, 2.0, 2.0]])).all()
            assert (trails[1] == np.array([[7.0, 7.0, 7.0],
                                           [8.0, 8.0, 8.0]])).all()

            trails = frame.parallel_trails_arrays_from_frame(rows=(1, 3))

            assert (trails[0] == np.array([[0.0, 0.0, 0.0],
                                           [1.0, 1.0, 1.0]])).all()
            assert (trails[1] == np.array([[6.0, 6.0, 6.0],
                                           [7.0, 7.0, 7.0]])).all()

    class TestExtractSerialFrontEdges:

        def test__pattern_bottom___extracts_1_front_edge_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #       /| Front Edge

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edge = frame.serial_front_edge_arrays_from_frame(columns=(0, 1))

            assert (front_edge == np.array([[1.0],
                                            [1.0],
                                            [1.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(columns=(1, 2))

            assert (front_edge == np.array([[2.0],
                                            [2.0],
                                            [2.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(columns=(2, 3))

            assert (front_edge == np.array([[3.0],
                                            [3.0],
                                            [3.0]])).all()

        def test__pattern_bottom___extracts_multiple_front_edges_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 5)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                    /| Front Edge

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edge = frame.serial_front_edge_arrays_from_frame(columns=(0, 2))

            assert (front_edge == np.array([[1.0, 2.0],
                                            [1.0, 2.0],
                                            [1.0, 2.0]])).all()

            front_edge = frame.serial_front_edge_arrays_from_frame(columns=(1, 4))

            assert (front_edge == np.array([[2.0, 3.0, 4.0],
                                            [2.0, 3.0, 4.0],
                                            [2.0, 3.0, 4.0]])).all()

        def test__pattern_bottom__2_regions__extracts_columns_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                    /| FE 1        /\ FE 2

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(0, 1))

            assert (front_edges[0] == np.array([[1.0],
                                                [1.0],
                                                [1.0]])).all()
            assert (front_edges[1] == np.array([[5.0],
                                                [5.0],
                                                [5.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(1, 2))

            assert (front_edges[0] == np.array([[2.0],
                                                [2.0],
                                                [2.0]])).all()
            assert (front_edges[1] == np.array([[6.0],
                                                [6.0],
                                                [6.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(2, 3))

            assert (front_edges[0] == np.array([[3.0],
                                                [3.0],
                                                [3.0]])).all()
            assert (front_edges[1] == np.array([[7.0],
                                                [7.0],
                                                [7.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(0, 3))

            assert (front_edges[0] == np.array([[1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0]])).all()

            assert (front_edges[1] == np.array([[5.0, 6.0, 7.0],
                                                [5.0, 6.0, 7.0],
                                                [5.0, 6.0, 7.0]])).all()

        def test__pattern_right__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                               /| FE 1            /\ FE 2

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(0, 1))

            assert (front_edges[0] == np.array([[3.0],
                                                [3.0],
                                                [3.0]])).all()
            assert (front_edges[1] == np.array([[7.0],
                                                [7.0],
                                                [7.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(1, 2))

            assert (front_edges[0] == np.array([[2.0],
                                                [2.0],
                                                [2.0]])).all()
            assert (front_edges[1] == np.array([[6.0],
                                                [6.0],
                                                [6.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(2, 3))

            assert (front_edges[0] == np.array([[1.0],
                                                [1.0],
                                                [1.0]])).all()
            assert (front_edges[1] == np.array([[5.0],
                                                [5.0],
                                                [5.0]])).all()

            front_edges = frame.serial_front_edge_arrays_from_frame(columns=(0, 3))

            assert (front_edges[0] == np.array([[1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0],
                                                [1.0, 2.0, 3.0]])).all()

            assert (front_edges[1] == np.array([[5.0, 6.0, 7.0],
                                                [5.0, 6.0, 7.0],
                                                [5.0, 6.0, 7.0]])).all()

    class TestExtractSerialTrails:

        def test__pattern_bottom___extracts_1_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                                    /| Trails Begin          

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 1))

            assert (trails == np.array([[4.0],
                                        [4.0],
                                        [4.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(1, 2))

            assert (trails == np.array([[5.0],
                                        [5.0],
                                        [5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(2, 3))

            assert (trails == np.array([[6.0],
                                        [6.0],
                                        [6.0]])).all()

        def test__pattern_bottom___extracts_multiple_trails_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                                   /| Trails Begin

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 2))

            assert (trails == np.array([[4.0, 5.0],
                                        [4.0, 5.0],
                                        [4.0, 5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(1, 4))

            assert (trails == np.array([[5.0, 6.0, 7.0],
                                        [5.0, 6.0, 7.0],
                                        [5.0, 6.0, 7.0]])).all()

        def test__pattern_bottom__2_regions__extracts_columns_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4), (0, 3, 5, 8)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

            #                                   /| Trails1           /\ Trails2

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 1))

            assert (trails[0] == np.array([[4.0],
                                           [4.0],
                                           [4.0]])).all()
            assert (trails[1] == np.array([[8.0],
                                           [8.0],
                                           [8.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(1, 2))

            assert (trails[0] == np.array([[5.0],
                                           [5.0],
                                           [5.0]])).all()
            assert (trails[1] == np.array([[9.0],
                                           [9.0],
                                           [9.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(2, 3))

            assert (trails[0] == np.array([[6.0],
                                           [6.0],
                                           [6.0]])).all()
            assert (trails[1] == np.array([[10.0],
                                           [10.0],
                                           [10.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 3))

            assert (trails[0] == np.array([[4.0, 5.0, 6.0],
                                           [4.0, 5.0, 6.0],
                                           [4.0, 5.0, 6.0]])).all()

            assert (trails[1] == np.array([[8.0, 9.0, 10.0],
                                           [8.0, 9.0, 10.0],
                                           [8.0, 9.0, 10.0]])).all()

        def test__pattern_right__does_all_the_above_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 3, 6), (0, 3, 8, 11)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

            #               Trails1   /|                Trails2 /\

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_right(), ci_pattern=pattern,
                                     array=image)

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 1))

            assert (trails[0] == np.array([[2.0],
                                           [2.0],
                                           [2.0]])).all()
            assert (trails[1] == np.array([[7.0],
                                           [7.0],
                                           [7.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(1, 2))

            assert (trails[0] == np.array([[1.0],
                                           [1.0],
                                           [1.0]])).all()
            assert (trails[1] == np.array([[6.0],
                                           [6.0],
                                           [6.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(2, 3))

            assert (trails[0] == np.array([[0.0],
                                           [0.0],
                                           [0.0]])).all()
            assert (trails[1] == np.array([[5.0],
                                           [5.0],
                                           [5.0]])).all()

            trails = frame.serial_trails_arrays_from_frame(columns=(0, 3))

            assert (trails[0] == np.array([[0.0, 1.0, 2.0],
                                           [0.0, 1.0, 2.0],
                                           [0.0, 1.0, 2.0]])).all()

            assert (trails[1] == np.array([[5.0, 6.0, 7.0],
                                           [5.0, 6.0, 7.0],
                                           [5.0, 6.0, 7.0]])).all()

    class TestParallelSerialCalibrationSection:

        def test__extracts_everything_except_prescan(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

            image = np.array([[0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            frame.frame_geometry.serial_prescan = ci_frame.Region(region=(0, 4, 0, 1))

            extracted_array = frame.parallel_serial_calibration_section()

            assert (extracted_array == np.array([[1.0, 2.0, 3.0],
                                                 [1.0, 2.0, 3.0],
                                                 [1.0, 2.0, 3.0],
                                                 [1.0, 2.0, 3.0],
                                                 [1.0, 2.0, 3.0]])).all()

        def test__extracts_everything_except_prescan_2(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

            image = np.array([[0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0],
                              [0.0, 1.0, 2.0, 3.0]])

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            frame.frame_geometry.serial_prescan = ci_frame.Region(region=(0, 4, 0, 2))

            extracted_array = frame.parallel_serial_calibration_section()

            assert (extracted_array == np.array([[2.0, 3.0],
                                                 [2.0, 3.0],
                                                 [2.0, 3.0],
                                                 [2.0, 3.0],
                                                 [2.0, 3.0]])).all()

    class TestMaskOnlyContainingSerialTrails:

        def test__pattern_bottom___mask_only_contains_trails(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                                    /| Trails Begin

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            mask = frame.mask_containing_only_serial_trails()

            assert type(mask) == msk.Mask
            assert mask.frame_geometry == frame.frame_geometry
            assert (mask == np.array([[True, True, True, True, False, False, False, False, False, False],
                                      [True, True, True, True, False, False, False, False, False, False],
                                      [True, True, True, True, False, False, False, False, False, False]])).all()

        def test__pattern_bottom__2_regions__extracts_columns_correctly(self):
            pattern = ci_pattern.CIPatternUniform(normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)])

            image = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

            #                                   /| Trails1           /\ Trails2

            frame = ci_frame.CIFrame(frame_geometry=ci_frame.QuadGeometryEuclid.bottom_left(), ci_pattern=pattern,
                                     array=image)

            mask = frame.mask_containing_only_serial_trails()

            assert type(mask) == msk.Mask
            assert mask.frame_geometry == frame.frame_geometry
            assert (mask == np.array([[True, True, True, True, False, False, False, False, False, False],
                                      [True, True, True, True, True, True, True, True, True, True],
                                      [True, True, True, True, False, False, False, False, False, False]])).all()


class TestQuadGeometryEuclid_bottom_left(object):
    class TestParallelFrontEdgeRegion:

        def test__extract_one_row__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is row 0, so for 1 row we extract 0 -> 1

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(0, 1))

            assert ci_front_edge == (0, 1, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is row 0, so for 2 rows we extract 0 -> 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(0, 2))

            assert ci_front_edge == (0, 2, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is row 0, so for these 2 rows we extract 1 ->2

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(1, 3))

            assert ci_front_edge == (1, 3, 0, 3)

    class TestParallelTrailsRegion:

        def test__extract_one_row__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # The trails are row 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(0, 1))

            assert ci_trails == (3, 4, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are row 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(0, 2))

            assert ci_trails == (3, 5, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are row 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(1, 3))

            assert ci_trails == (4, 6, 0, 3)

    class TestParallelSideNearestReadOut:

        def test__columns_0_to_1__region_is_left_hand_side(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region, image_shape=(5, 5),
                                                                                   columns=(0, 1))

            assert ci_parallel_region == (0, 5, 0, 1)

        def test__columns_1_to_3__region_is_left_hand_side(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(4, 4), columns=(1, 3))

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_1_to_3__different_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 2, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(4, 4), columns=(1, 3))

            assert ci_parallel_region == (0, 4, 3, 5)

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(2, 5), columns=(0, 1))

            assert ci_parallel_region == (0, 2, 0, 1)

    class TestSerialFrontEdgeRegion:

        def test__extract_one_column__takes_coordinates_from_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is column 0, so for 1 column we extract 0 -> 1

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(0, 1))

            assert ci_front_edge == (0, 3, 0, 1)

        def test__extract_two_columns__first_and_second__takes_coordinates_from_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is column 0, so for 2 columns we extract 0 -> 2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(0, 2))

            assert ci_front_edge == (0, 3, 0, 2)

        def test__extract_two_columns__second_and_third__takes_coordinates_from_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(
                region=(0, 3, 0, 3))  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(1, 3))

            assert ci_front_edge == (0, 3, 1, 3)

    class TestSerialTrailsRegion:

        def test__extract_one_row__takes_coordinates_after_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # The trails are column 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(0, 1))

            assert ci_trails == (0, 3, 3, 4)

        def test__extract_two_columns__first_and_second__takes_coordinates_after_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are column 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(0, 2))

            assert ci_trails == (0, 3, 3, 5)

        def test__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(1, 3))

            assert ci_trails == (0, 3, 4, 6)

    class TestSerialChargeInjectionAndTrails:

        def test__column_0_of_front_edge__region_is_left_hand_side__no_overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 5),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 0, 5)

        def test__column_0_of_front_edge__region_is_left_hand_side__overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 0, 25)

        def test__column_2__region_is_left_hand_side__overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=2)

            assert ci_serial_region == (1, 3, 2, 25)

        def test__ci_region_has_overscan_and_prescan_either_side__prescan_ignored(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 10, 20))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 10, 25)

        def test__ci_region_has_overscan_and_prescan_either_side__include_column(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 10, 20))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=2)

            assert ci_serial_region == (1, 3, 12, 25)

        def test__different_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(8, 55),
                                                                       column=2)

            assert ci_serial_region == (3, 5, 7, 55)


class TestQuadGeometryEuclid_bottom_right(object):
    class TestParallelFrontEdgeOfRegion:

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is row 0, so for these 2 rows we extract 1 ->2

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(1, 3))

            assert ci_front_edge == (1, 3, 0, 3)

    class TestParallelTrailsRegion:

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are row 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(1, 3))

            assert ci_trails == (4, 6, 0, 3)

    class TestParallelSideNearestReadOut:

        def test__columns_0_to_1__region_is_right_hand_side(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(5, 5), columns=(0, 1))

            assert ci_parallel_region == (0, 5, 4, 5)

        def test__columns_1_to_3__region_is_right_hand_side(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 0, 4))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(4, 4), columns=(1, 3))

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_1_to_3__different_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 2, 4))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(4, 4), columns=(1, 3))

            assert ci_parallel_region == (0, 4, 1, 3)

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(0, 1, 0, 5))
            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(2, 5), columns=(0, 1))

            assert ci_parallel_region == (0, 2, 4, 5)

    class TestSerialFrontEdgeRegion:

        def test__extract_one_column__takes_coordinates_from_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is column 0, so for 1 column we extract 0 -> 1

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(0, 1))

            assert ci_front_edge == (0, 3, 2, 3)

        def test__extract_two_columns__first_and_second__takes_coordinates_from_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(region=(0, 3, 0, 3))  # Front edge is column 0, so for 2 columns we extract 0 -> 2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(0, 2))

            assert ci_front_edge == (0, 3, 1, 3)

        def test__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(
                region=(0, 3, 0, 3))  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(1, 3))

            assert ci_front_edge == (0, 3, 0, 2)

    class TestSerialTrailsRegion:

        def test__extract_one_row__takes_coordinates_after_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(region=(0, 3, 3, 6))  # The trails are column 3 and above, so extract 3 -> 4

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(0, 1))

            assert ci_trails == (0, 3, 2, 3)

        def test__extract_two_columns__first_and_second__takes_coordinates_after_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion((0, 3, 3, 6))  # The trails are column 3 and above, so extract 3 -> 5

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(0, 2))

            assert ci_trails == (0, 3, 1, 3)

        def test__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion((0, 3, 3, 6))  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(1, 3))

            assert ci_trails == (0, 3, 0, 2)

    class TestSerialChargeInjectionAndTrails:

        def test__column_0_of_front_edge__region_is_left_hand_side__no_overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 5),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 0, 5)

        def test__column_0_of_front_edge__region_is_left_hand_side__overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 20, 25))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 0, 25)

        def test__column_2__region_is_left_hand_side__overscan_beyond_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 20, 25))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=2)

            assert ci_serial_region == (1, 3, 0, 23)

        def test__ci_region_has_overscan_and_prescan_either_side__prescan_ignored(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 10, 20))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=0)

            assert ci_serial_region == (1, 3, 0, 20)

        def test__ci_region_has_overscan_and_prescan_either_side__include_column(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(1, 3, 10, 20))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(5, 25),
                                                                       column=2)

            assert ci_serial_region == (1, 3, 0, 18)

        def test__different_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
            ci_region = MockRegion(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(8, 55),
                                                                       column=2)

            assert ci_serial_region == (3, 5, 0, 28)


class TestQuadGeometryEuclid_top_left(object):
    class TestParallelFrontEdgeOfRegion:

        def test__extract_one_row__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The front edge is closest to 3, so for 1 edge we extract row 3-> 4

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(0, 1))

            assert ci_front_edge == (2, 3, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                (0, 3, 0, 3))  # The front edge is closest to 3, so for these 2 rows we extract 2 & 3

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(0, 2))

            assert ci_front_edge == (1, 3, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                (0, 3, 0, 3))  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(1, 3))

            assert ci_front_edge == (0, 2, 0, 3)

    class TestParallelTrailsRegion:

        def test__extract_one_row__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                (3, 5, 0, 3))  # The trails are the rows after row 3, so for 1 edge we should extract just row 2

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(0, 1))

            assert ci_trails == (2, 3, 0, 3)

        def test__extract_two_rows__first_and_second__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                (3, 5, 0, 3))  # The trails are the row after row 3, so for these 2 edges we extract rows 1->3

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(0, 2))

            assert ci_trails == (1, 3, 0, 3)

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                (3, 5, 0, 3))  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(1, 3))

            assert ci_trails == (0, 2, 0, 3)

    class TestParallelSideNearestReadOut:

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
            ci_region = MockRegion(region=(1, 3, 0, 5))

            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(2, 5), columns=(0, 1))

            assert ci_parallel_region == (0, 2, 0, 1)

    class TestSerialFrontEdgeRegion:

        def test__extract_two_columns__second_and_third__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion(
                region=(0, 3, 0, 3))  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(1, 3))

            assert ci_front_edge == (0, 3, 1, 3)

    class TestSerialTrailsRegion:

        def test__extract_two_columns__second_and_third__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()

            ci_region = MockRegion((0, 3, 0, 3))  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(1, 3))

            assert ci_trails == (0, 3, 4, 6)

    class TestSerialChargeInjectionAndTrails:

        def test__different_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_left()
            ci_region = MockRegion(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(8, 55),
                                                                       column=2)

            assert ci_serial_region == (3, 5, 7, 55)


class TestQuadGeometryEuclid_top_right(object):
    class TestParallelFrontEdgeOfRegion:

        def test__extract_two_rows__second_and_third__takes_coordinates_from_top_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_right()

            ci_region = MockRegion(
                (0, 3, 0, 3))  # The front edge is closest to 3, so for these 2 rows we extract 1 & 2

            ci_front_edge = ci_geometry.parallel_front_edge_region(region=ci_region, rows=(1, 3))

            assert ci_front_edge == (0, 2, 0, 3)

    class TestParallelTrailsRegion:

        def test__extract_two_rows__second_and_third__takes_coordinates_after_bottom_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_right()

            ci_region = MockRegion(
                (3, 5, 0, 3))  # The trails are the row after row 3, so for these 2 edges we extract rows 0 & 2

            ci_trails = ci_geometry.parallel_trails_region(region=ci_region, rows=(1, 3))

            assert ci_trails == (0, 2, 0, 3)

    class TestParallelSideNearestReadOut:

        def test__columns_0_to_1__asymetric_image(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_right()
            ci_region = MockRegion(region=(0, 1, 0, 5))
            ci_parallel_region = ci_geometry.parallel_side_nearest_read_out_region(region=ci_region,
                                                                                   image_shape=(2, 5), columns=(0, 1))

            assert ci_parallel_region == (0, 2, 4, 5)

    class TestSerialFrontEdgeRegion:

        def test__extract_two_columns__second_and_third__takes_coordinates_from_right_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion(
                region=(0, 3, 0, 3))  # Front edge is column 0, so for these 2 columns we extract 1 ->2

            ci_front_edge = ci_geometry.serial_front_edge_region(region=ci_region, columns=(1, 3))

            assert ci_front_edge == (0, 3, 0, 2)

    class TestSerialTrailsRegion:

        def test__extract_two_columns__second_and_third__takes_coordinates_after_left_of_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.bottom_right()

            ci_region = MockRegion((0, 3, 3, 6))  # The trails are column 3 and above, so extract 4 -> 6

            ci_trails = ci_geometry.serial_trails_region(region=ci_region, columns=(1, 3))

            assert ci_trails == (0, 3, 0, 2)

    class TestSerialChargeInjectionAndTrails:

        def test__different_ci_region(self):
            ci_geometry = ci_frame.QuadGeometryEuclid.top_right()
            ci_region = MockRegion(region=(3, 5, 5, 30))

            ci_serial_region = ci_geometry.serial_ci_region_and_trails(region=ci_region, image_shape=(8, 55),
                                                                       column=2)

            assert ci_serial_region == (3, 5, 0, 28)
