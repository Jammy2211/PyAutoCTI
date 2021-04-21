import numpy as np
import autocti as ac


class TestSerialEdgesAndTrailsFrame:
    def test__front_edge_only__multiple_columns__new_frame_is_only_edge(self):
        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 3, 0, 3)])

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        new_array = layout.serial_edges_and_trails_frame(front_edge_columns=(0, 1))

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
            )
        ).all()

        new_array = layout.serial_edges_and_trails_frame(front_edge_columns=(0, 2))

        assert (
            new_array
            == np.array(
                [[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 3, 0, 3)]

    def test__trails_only__multiple_columns__new_frame_is_only_that_trail(self):
        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 3, 0, 2)])

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        new_array = layout.serial_edges_and_trails_frame(trails_columns=(0, 1))

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]]
            )
        ).all()

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        new_array = layout.serial_edges_and_trails_frame(trails_columns=(0, 2))

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 3, 0, 2)]

    def test__front_edge_and_trails__2_columns_of_each__new_frame_is_edge_and_trail(
        self,
    ):
        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 3, 0, 2)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        new_array = layout.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 2)
        )

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 1.1, 2.0, 0.0],
                    [4.0, 0.0, 1.1, 6.0, 0.0],
                    [8.0, 0.0, 1.1, 10.0, 0.0],
                ]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 3, 0, 2)]

    def test__front_edge_and_trails__2_region_list_1_column_of_each__new_frame_is_edge_and_trail(
        self,
    ):
        layout = ac.ci.Layout2DCIUniform(
            normalization=10.0, region_list=[(0, 3, 0, 1), (0, 3, 3, 4)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        new_array = layout.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 1)
        )

        assert (
            new_array
            == np.array(
                [
                    [0.0, 1.0, 0.0, 2.0, 3.0],
                    [4.0, 5.0, 0.0, 6.0, 7.0],
                    [8.0, 9.0, 0.0, 10.0, 11.0],
                ]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 3, 0, 1), (0, 3, 3, 4)]


class TestSerialAllTrailsFrame:
    def test__1_ci_region__multiple_serial_trail__extracts_all_trails(self):
        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 4, 0, 2)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_overscan=(0, 4, 2, 3)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_trails_frame_from

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 4, 0, 2)]

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_overscan=(0, 4, 2, 4)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_trails_frame_from

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 2.0, 0.5],
                    [0.0, 0.0, 5.0, 0.5],
                    [0.0, 0.0, 8.0, 0.5],
                    [0.0, 0.0, 11.0, 0.5],
                ]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 4, 0, 2)]

    def test__2_region_list_ci__2_serial_trail__extracts_all_trails(self):
        layout = ac.ci.Layout2DCIUniform(
            normalization=10.0, region_list=[(0, 1, 0, 2), (2, 3, 0, 2)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_overscan=(0, 4, 2, 4)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_trails_frame_from

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 2.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(0, 1, 0, 2), (2, 3, 0, 2)]


class TestSerialOverScanAboveTrailsFrame:
    def test__1_ci_region__serial_trails_go_over_hand_columns(self):
        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(1, 3, 1, 2)])

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1), serial_overscan=(0, 3, 2, 4)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_overscan_no_trails_frame_from

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(1, 3, 1, 2)]

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(1, 3, 1, 3)])

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1), serial_overscan=(0, 3, 3, 4)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_overscan_no_trails_frame_from

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_layout.layout_ci.region_list == [(1, 3, 1, 3)]

    def test__2_region_list_ci__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
        self,
    ):
        layout = ac.ci.Layout2DCIUniform(
            normalization=10.0, region_list=[(1, 2, 1, 3), (3, 4, 1, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 5, 0, 1), serial_overscan=(0, 5, 3, 4)),
            pixel_scales=1.0,
        )

        new_array = layout.serial_overscan_no_trails_frame_from

        assert (
            new_array
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
        assert new_layout.layout_ci.region_list == [(1, 2, 1, 3), (3, 4, 1, 3)]


class TestSerialCalibrationFrame:
    def test__ci_region_across_all_image__column_0__extracts_all_columns(self):

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 3, 1, 5)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1)),
            pixel_scales=1.0,
        )

        serial_frame = layout.serial_calibration_frame_from(rows=(0, 3))

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

        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.layout_ci.region_list == [(0, 3, 1, 5)]
        assert serial_frame.layout.parallel_overscan == None
        assert serial_frame.layout.serial_prescan == (0, 3, 0, 1)
        assert serial_frame.layout.serial_overscan == None
        assert serial_frame.pixel_scales == (1.0, 1.0)

    def test__2_region_list_ci__both_extracted(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 1, 1, 3), (2, 3, 1, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1), serial_overscan=(0, 3, 3, 4)),
            pixel_scales=1.0,
        )

        serial_frame = layout.serial_calibration_frame_from(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.layout_ci.region_list == [(0, 1, 1, 3), (1, 2, 1, 3)]
        assert serial_frame.layout.parallel_overscan == None
        assert serial_frame.layout.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.layout.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == (1.0, 1.0)

    def test__rows_cuts_out_bottom_row(self):

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 2, 1, 4)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1), serial_overscan=(0, 3, 3, 4)),
            pixel_scales=1.0,
        )

        serial_frame = layout.serial_calibration_frame_from(rows=(0, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.layout_ci.region_list == [(0, 2, 1, 4)]
        assert serial_frame.layout.parallel_overscan == None
        assert serial_frame.layout.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.layout.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == (1.0, 1.0)

    def test__extract_two_region_list_and_cut_bottom_row_from_each(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 5, 0, 1), serial_overscan=(0, 5, 3, 4)),
            pixel_scales=1.0,
        )

        serial_frame = layout.serial_calibration_frame_from(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 5.0, 5.0, 5.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.layout_ci.region_list == [(0, 1, 1, 4), (1, 2, 1, 4)]
        assert serial_frame.layout.parallel_overscan == None
        assert serial_frame.layout.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.layout.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == (1.0, 1.0)

    def test__extract_two_region_list_and_cut_top_row_from_each(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(serial_prescan=(0, 3, 0, 1), serial_overscan=(0, 3, 3, 4)),
            pixel_scales=1.0,
        )

        serial_frame = layout.serial_calibration_frame_from(rows=(1, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.layout_ci.region_list == [(0, 1, 1, 4), (1, 2, 1, 4)]
        assert serial_frame.layout.parallel_overscan == None
        assert serial_frame.layout.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.layout.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == (1.0, 1.0)

    def test__extract_mask_as_above(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 5), pixel_scales=1.0)

        mask[1, 1] = True
        mask[4, 3] = True

        serial_frame = layout.serial_calibration_mask_from_mask_and_rows(
            mask=mask, rows=(1, 2)
        )

        assert (
            serial_frame
            == np.array(
                [[False, True, False, False, False], [False, False, False, True, False]]
            )
        ).all()


class TestSerialCalibrationArrays:
    def test__different_region_list_ci__extracts_all_columns(self):
        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 3, 0, 5)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        serial_region = layout.serial_calibration_sub_arrays_from

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

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 3, 0, 4)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        serial_region = layout.serial_calibration_sub_arrays_from

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

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 3, 1, 4)])

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        serial_region = layout.serial_calibration_sub_arrays_from

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

    def test__2_region_list_ci__both_extracted(self):
        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        serial_region = layout.serial_calibration_sub_arrays_from

        assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
        assert (serial_region[1] == np.array([[0.0, 1.0, 4.0, 4.0, 4.0]])).all()


class TestSerialTrailsArrays:
    def test__1_region__extracts_multiple_trails_correctly(self):
        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 3, 1, 4)])

        #                                    /| Trails Begin

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        trails = layout.serial_trails_arrays_from(columns=(0, 1))

        assert (trails == np.array([[4.0], [4.0], [4.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(1, 2))

        assert (trails == np.array([[5.0], [5.0], [5.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(2, 3))

        assert (trails == np.array([[6.0], [6.0], [6.0]])).all()

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        trails = layout.serial_trails_arrays_from(columns=(0, 2))

        assert (trails == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(1, 4))

        assert (
            trails == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__2_region_list__extracts_columns_correctly(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        #                                   /| Trails1           /\ Trails2

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        trails = layout.serial_trails_arrays_from(columns=(0, 1))

        assert (trails[0] == np.array([[4.0], [4.0], [4.0]])).all()
        assert (trails[1] == np.array([[8.0], [8.0], [8.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(1, 2))

        assert (trails[0] == np.array([[5.0], [5.0], [5.0]])).all()
        assert (trails[1] == np.array([[9.0], [9.0], [9.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(2, 3))

        assert (trails[0] == np.array([[6.0], [6.0], [6.0]])).all()
        assert (trails[1] == np.array([[10.0], [10.0], [10.0]])).all()

        trails = layout.serial_trails_arrays_from(columns=(0, 3))

        assert (
            trails[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        ).all()

        assert (
            trails[1]
            == np.array([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]

        #                                   /| Trails1           /\ Trails2

        mask = ac.Mask2D.manual(
            mask=[
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
            ],
            pixel_scales=1.0,
        )

        array = ac.Array2D.manual_mask(array=arr, mask=mask, layout_ci=layout)

        trails = layout.serial_trails_arrays_from(columns=(0, 3))

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

    def test__stacked_array_and_binned_line__2_region_list__no_masking(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        #                                   /| Trails1           /\ Trails2

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ],
            layout_ci=layout,
            pixel_scales=1.0,
        )

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = layout.serial_trails_stacked_array_from(columns=(0, 3))

        assert (
            stacked_trails
            == np.array([[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 7.0, 8.0]])
        ).all()

        trails_line = layout.serial_trails_line_binned_over_rows_from(columns=(0, 3))

        assert (trails_line == np.array([6.0, 7.0, 8.0])).all()

    def test__same_as_above__include_masking(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        arr = [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]

        #                                      /| Trails1           /\ Trails2

        mask = ac.Mask2D.manual(
            mask=[
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
            ],
            pixel_scales=1.0,
        )

        #                                               /| Trails1                   /\ Trails2

        array = ac.Array2D.manual_mask(array=arr, mask=mask, layout_ci=layout)

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = layout.serial_trails_stacked_array_from(columns=(0, 3))

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

        trails_line = layout.serial_trails_line_binned_over_rows_from(columns=(0, 3))

        assert (trails_line == np.array([20.0 / 3.0, 19.0 / 3.0, 26.0 / 3.0])).all()

        mask = ac.Mask2D.manual(
            mask=[
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
            ],
            pixel_scales=1.0,
        )

        array = ac.Array2D.manual_mask(array=arr, mask=mask, layout_ci=layout)

        stacked_trails = layout.serial_trails_stacked_array_from(columns=(0, 3))

        assert (
            stacked_trails.mask
            == np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            )
        ).all()

    def test__no_columns_specified_so_uses_full_serial_overscan(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[(0, 3, 1, 4), (0, 3, 4, 7)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            ],
            layout_ci=layout,
            scans=ac.Scans(
                serial_overscan=ac.Region2D((0, 1, 0, 4)),
                serial_prescan=ac.Region2D((0, 1, 0, 1)),
                parallel_overscan=ac.Region2D((0, 1, 0, 1)),
            ),
            pixel_scales=1.0,
        )

        trails = layout.serial_trails_arrays_from()

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

        stacked_trails = layout.serial_trails_stacked_array_from()

        assert (
            stacked_trails
            == np.array(
                [[5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5], [5.5, 6.5, 7.5, 8.5]]
            )
        ).all()

        trails_line = layout.serial_trails_line_binned_over_rows_from()

        assert (trails_line == np.array([5.5, 6.5, 7.5, 8.5])).all()


class TestExtractions:
    def test__parallel_serial_calibration_section__extracts_everything(self):

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 1, 0, 1)])

        arr = [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]

        array = ac.Array2D.manual(array=arr, layout_ci=layout, pixel_scales=1.0)

        assert (layout.parallel_serial_calibration_frame == arr).all()

        layout = ac.ci.Layout2DCIUniform(normalization=1.0, region_list=[(0, 1, 0, 1)])

        array = ac.Array2D.manual(array=arr, layout_ci=layout, pixel_scales=1.0)

        assert (layout.parallel_serial_calibration_frame == arr).all()

    def test__smallest_parallel_trails_rows_to_frame_edge__x2_ci_region__bottom_frame_geometry(
        self,
    ):

        layout = ac.ci.Layout2DCIUniform(
            normalization=10.0, region_list=[(0, 3, 0, 3), (5, 7, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=np.ones((10, 5)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.smallest_parallel_trails_rows_to_frame_edge == 2

        array = ac.Array2D.manual(
            array=np.ones((8, 5)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.smallest_parallel_trails_rows_to_frame_edge == 1

    def test__serial_trails_columns__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
        self, layout_ci_7x7
    ):

        array = ac.Array2D.manual(
            array=np.ones((10, 10)),
            layout_ci=layout_ci_7x7,
            scans=ac.Scans(
                serial_overscan=ac.Region2D((0, 1, 0, 10)),
                serial_prescan=ac.Region2D((0, 1, 0, 1)),
                parallel_overscan=ac.Region2D((0, 1, 0, 1)),
            ),
            pixel_scales=1.0,
        )

        assert layout.layout.serial_trails_columns == 10

        array = ac.Array2D.manual(
            array=np.ones((50, 50)),
            layout_ci=layout_ci_7x7,
            scans=ac.Scans(
                serial_overscan=ac.Region2D((0, 1, 0, 50)),
                serial_prescan=ac.Region2D((0, 1, 0, 1)),
                parallel_overscan=ac.Region2D((0, 1, 0, 1)),
            ),
            pixel_scales=1.0,
        )

        assert layout.layout.serial_trails_columns == 50

    def test__parallel_trail_size_to_frame_edge__parallel_trail_size_to_edge(self):

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0, region_list=[ac.Region2D(region=(0, 3, 0, 3))]
        )

        array = ac.Array2D.manual(
            array=np.ones((5, 100)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.parallel_trail_size_to_frame_edge == 2

        array = ac.Array2D.manual(
            array=np.ones((7, 100)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.parallel_trail_size_to_frame_edge == 4

        layout = ac.ci.Layout2DCIUniform(
            normalization=1.0,
            region_list=[
                ac.Region2D(region=(0, 2, 0, 3)),
                ac.Region2D(region=(5, 8, 0, 3)),
                ac.Region2D(region=(11, 14, 0, 3)),
            ],
        )

        array = ac.Array2D.manual(
            array=np.ones((15, 100)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.parallel_trail_size_to_frame_edge == 1

        array = ac.Array2D.manual(
            array=np.ones((20, 100)), layout_ci=layout, pixel_scales=1.0
        )

        assert layout.parallel_trail_size_to_frame_edge == 6


class TestCIFrameAPI:
    def test__manual__makes_array_using_inputs(self):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        array = ac.Array2D.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()
        assert layout.native.layout_ci.region_list == [(0, 1, 0, 1)]

        array = ac.Array2D.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            layout_ci=layout,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
        assert layout.layout_ci.region_list == [(1, 2, 0, 1)]
        assert layout.original_roe_corner == (0, 0)
        assert layout.layout.parallel_overscan == (1, 2, 0, 1)
        assert layout.layout.serial_prescan == (0, 1, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

        array = ac.Array2D.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            layout_ci=layout,
            roe_corner=(1, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 1, 2)]
        assert layout.original_roe_corner == (1, 1)
        assert layout.layout.parallel_overscan == (0, 1, 1, 2)
        assert layout.layout.serial_prescan == (1, 2, 0, 1)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

        array = ac.Array2D.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            layout_ci=layout,
            roe_corner=(0, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
        assert layout.layout_ci.region_list == [(1, 2, 1, 2)]
        assert layout.original_roe_corner == (0, 1)
        assert layout.layout.parallel_overscan == (1, 2, 1, 2)
        assert layout.layout.serial_prescan == (0, 1, 0, 1)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

    def test__full_ones_zeros__makes_frame_using_inputs(self):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 3, 0, 3)])

        array = ac.Array2D.full(
            fill_value=8.0,
            shape_native=(2, 2),
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
        assert layout.layout_ci.region_list == [(0, 3, 0, 3)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

        array = ac.Array2D.ones(
            shape_native=(2, 2),
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert layout.layout_ci.region_list == [(0, 3, 0, 3)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

        array = ac.Array2D.zeros(
            shape_native=(2, 2),
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        assert (array == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert layout.layout_ci.region_list == [(0, 3, 0, 3)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

    def test__extracted_array_from_array_and_extraction_region(self):

        array = ac.Array2D.manual(
            array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            layout_ci=ac.ci.Layout2DCIUniform(
                region_list=[(0, 1, 0, 1), (1, 2, 1, 2)], normalization=10.0
            ),
            scans=ac.Scans(
                parallel_overscan=None,
                serial_prescan=(0, 2, 0, 2),
                serial_overscan=(1, 2, 1, 2),
            ),
            pixel_scales=1.0,
        )

        array = ac.Array2D.extracted_array_from_array_and_extraction_region(
            array=array, extraction_region=ac.Region2D(region=(1, 3, 1, 3))
        )

        assert (array == np.array([[5.0, 6.0], [8.0, 9.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == None
        assert layout.layout.serial_prescan == (0, 1, 0, 1)
        assert layout.layout.serial_overscan == (0, 1, 0, 1)
        assert (layout.mask == np.array([[False, False], [False, False]])).all()

    def test__manual_mask__makes_array_using_inputs(self):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        mask = ac.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        array = ac.Array2D.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (array == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, True], [False, False]])).all()

        array = ac.Array2D.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            layout_ci=layout,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (array == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert layout.layout_ci.region_list == [(1, 2, 0, 1)]
        assert layout.original_roe_corner == (0, 0)
        assert layout.layout.parallel_overscan == (1, 2, 0, 1)
        assert layout.layout.serial_prescan == (0, 1, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [False, True]])).all()

        array = ac.Array2D.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            layout_ci=layout,
            roe_corner=(1, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (array == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 1, 2)]
        assert layout.original_roe_corner == (1, 1)
        assert layout.layout.parallel_overscan == (0, 1, 1, 2)
        assert layout.layout.serial_prescan == (1, 2, 0, 1)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[True, False], [False, False]])).all()

        array = ac.Array2D.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            layout_ci=layout,
            roe_corner=(0, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (array == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert layout.layout_ci.region_list == [(1, 2, 1, 2)]
        assert layout.original_roe_corner == (0, 1)
        assert layout.layout.parallel_overscan == (1, 2, 1, 2)
        assert layout.layout.serial_prescan == (0, 1, 0, 1)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, False], [True, False]])).all()

    def test__from_array__from_frame__makes_frame_using_inputs(self):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        mask = ac.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        array = ac.Array2D.full(
            shape_native=(2, 2),
            fill_value=8.0,
            layout_ci=layout,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        array = ac.Array2D.from_array(array=array, mask=mask)

        assert (array == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert (layout.native == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert layout.original_roe_corner == (1, 0)
        assert layout.layout.parallel_overscan == (0, 1, 0, 1)
        assert layout.layout.serial_prescan == (1, 2, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, True], [False, False]])).all()

        array = ac.Array2D.full(
            shape_native=(2, 2),
            fill_value=8.0,
            layout_ci=layout,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
            pixel_scales=1.0,
        )

        array = ac.Array2D.from_array(array=array, mask=mask)

        assert (array == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert layout.layout_ci.region_list == [(1, 2, 0, 1)]
        assert layout.original_roe_corner == (0, 0)
        assert layout.layout.parallel_overscan == (1, 2, 0, 1)
        assert layout.layout.serial_prescan == (0, 1, 1, 2)
        assert layout.layout.serial_overscan == (0, 2, 0, 2)
        assert (layout.mask == np.array([[False, True], [False, False]])).all()


class TestCIFrameEuclid:
    def test__euclid_array_for_four_quandrants__loads_data_and_dimensions(
        self, euclid_data
    ):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        euclid_array = ac.Array2DEuclid.top_left(array=euclid_data, layout_ci=layout)

        assert isinstance(euclid_array, ac.Array2D)
        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 0, 1)]
        assert euclid_layout.original_roe_corner == (0, 0)
        assert euclid_layout.shape_native == (2086, 2128)
        assert (euclid_array == np.zeros((2086, 2128))).all()
        assert euclid_layout.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_layout.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_layout.layout.serial_overscan == (20, 2086, 2099, 2128)

        euclid_array = ac.Array2DEuclid.top_right(array=euclid_data, layout_ci=layout)

        assert isinstance(euclid_array, ac.Array2D)
        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (0, 1)
        assert euclid_layout.shape_native == (2086, 2128)
        assert (euclid_array == np.zeros((2086, 2128))).all()
        assert euclid_layout.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_layout.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_layout.layout.serial_overscan == (20, 2086, 2099, 2128)

        euclid_array = ac.Array2DEuclid.bottom_left(array=euclid_data, layout_ci=layout)

        assert isinstance(euclid_array, ac.Array2D)
        assert euclid_layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert euclid_layout.original_roe_corner == (1, 0)
        assert euclid_layout.shape_native == (2086, 2128)
        assert (euclid_array == np.zeros((2086, 2128))).all()
        assert euclid_layout.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_layout.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_layout.layout.serial_overscan == (0, 2066, 2099, 2128)

        euclid_array = ac.Array2DEuclid.bottom_right(
            array=euclid_data, layout_ci=layout
        )

        assert isinstance(euclid_array, ac.Array2D)
        assert euclid_layout.layout_ci.region_list == [(0, 1, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (1, 1)
        assert euclid_layout.shape_native == (2086, 2128)
        assert (euclid_array == np.zeros((2086, 2128))).all()
        assert euclid_layout.layout.parallel_overscan == (2066, 2086, 51, 2099)
        assert euclid_layout.layout.serial_prescan == (0, 2086, 0, 51)
        assert euclid_layout.layout.serial_overscan == (0, 2066, 2099, 2128)

    def test__left_side__chooses_correct_frame_given_input(self, euclid_data):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="E", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert euclid_layout.original_roe_corner == (1, 0)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="E", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert euclid_layout.original_roe_corner == (1, 0)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="E", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 0, 1)]
        assert euclid_layout.original_roe_corner == (1, 0)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="F", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (1, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="F", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (1, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="F", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(0, 1, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (1, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="G", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (0, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="G", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (0, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="G", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 2127, 2128)]
        assert euclid_layout.original_roe_corner == (0, 1)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text1", quadrant_id="H", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 0, 1)]
        assert euclid_layout.original_roe_corner == (0, 0)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text2", quadrant_id="H", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 0, 1)]
        assert euclid_layout.original_roe_corner == (0, 0)

        euclid_array = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text3", quadrant_id="H", layout_ci=layout
        )

        assert euclid_layout.layout_ci.region_list == [(2085, 2086, 0, 1)]
        assert euclid_layout.original_roe_corner == (0, 0)

    def test__right_side__chooses_correct_frame_given_input(self, euclid_data):

        layout = ac.ci.Layout2DCIUniform(normalization=10.0, region_list=[(0, 1, 0, 1)])

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="E", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="E", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="E", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 1)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="F", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="F", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="F", layout_ci=layout
        )

        assert frame.original_roe_corner == (0, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="G", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="G", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="G", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 0)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text4", quadrant_id="H", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text5", quadrant_id="H", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 1)

        frame = ac.Array2DEuclid.from_ccd_and_quadrant_id(
            array=euclid_data, ccd_id="text6", quadrant_id="H", layout_ci=layout
        )

        assert frame.original_roe_corner == (1, 1)
