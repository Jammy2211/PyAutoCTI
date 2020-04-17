import numpy as np

from autocti import structures as struct
from autocti import charge_injection as ci

from autocti.charge_injection import ci_mask as ci_msk


class TestCIFrameAPI:
    def test__manual__makes_ci_frame_using_inputs(self):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 1)])

        ci_frame = ci.CIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 0, 1)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

        ci_frame = ci.CIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            ci_pattern=pattern,
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
        assert ci_frame.ci_pattern.regions == [(1, 2, 0, 1)]
        assert ci_frame.original_roe_corner == (0, 0)
        assert ci_frame.parallel_overscan == (1, 2, 0, 1)
        assert ci_frame.serial_prescan == (0, 1, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

        ci_frame = ci.CIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            ci_pattern=pattern,
            roe_corner=(1, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 1, 2)]
        assert ci_frame.original_roe_corner == (1, 1)
        assert ci_frame.parallel_overscan == (0, 1, 1, 2)
        assert ci_frame.serial_prescan == (1, 2, 0, 1)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

        ci_frame = ci.CIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            ci_pattern=pattern,
            roe_corner=(0, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
        assert ci_frame.ci_pattern.regions == [(1, 2, 1, 2)]
        assert ci_frame.original_roe_corner == (0, 1)
        assert ci_frame.parallel_overscan == (1, 2, 1, 2)
        assert ci_frame.serial_prescan == (0, 1, 0, 1)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

    def test__full_ones_zeros__makes_frame_using_inputs(self):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        ci_frame = ci.CIFrame.full(
            fill_value=8.0,
            shape_2d=(2, 2),
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

        ci_frame = ci.CIFrame.ones(
            shape_2d=(2, 2),
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

        ci_frame = ci.CIFrame.zeros(
            shape_2d=(2, 2),
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()

    def test__extracted_ci_frame_from_ci_frame_and_extraction_region(self):

        ci_frame = ci.CIFrame.manual(
            array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            ci_pattern=ci.CIPatternUniform(
                regions=[(0, 1, 0, 1), (1, 2, 1, 2)], normalization=10.0
            ),
            roe_corner=(1, 0),
            parallel_overscan=None,
            serial_prescan=(0, 2, 0, 2),
            serial_overscan=(1, 2, 1, 2),
        )

        ci_frame = ci.CIFrame.extracted_ci_frame_from_ci_frame_and_extraction_region(
            ci_frame=ci_frame, extraction_region=struct.Region(region=(1, 3, 1, 3))
        )

        assert (ci_frame == np.array([[5.0, 6.0], [8.0, 9.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 0, 1)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == None
        assert ci_frame.serial_prescan == (0, 1, 0, 1)
        assert ci_frame.serial_overscan == (0, 1, 0, 1)
        assert (ci_frame.mask == np.array([[False, False], [False, False]])).all()


class TestCIMaskedFrameAPI:
    def test__manual__makes_ci_frame_using_inputs(self):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 1)])

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        ci_frame = ci.MaskedCIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 0, 1)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()

        ci_frame = ci.MaskedCIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert ci_frame.ci_pattern.regions == [(1, 2, 0, 1)]
        assert ci_frame.original_roe_corner == (0, 0)
        assert ci_frame.parallel_overscan == (1, 2, 0, 1)
        assert ci_frame.serial_prescan == (0, 1, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [False, True]])).all()

        ci_frame = ci.MaskedCIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(1, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 1, 2)]
        assert ci_frame.original_roe_corner == (1, 1)
        assert ci_frame.parallel_overscan == (0, 1, 1, 2)
        assert ci_frame.serial_prescan == (1, 2, 0, 1)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[True, False], [False, False]])).all()

        ci_frame = ci.MaskedCIFrame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(0, 1),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert ci_frame.ci_pattern.regions == [(1, 2, 1, 2)]
        assert ci_frame.original_roe_corner == (0, 1)
        assert ci_frame.parallel_overscan == (1, 2, 1, 2)
        assert ci_frame.serial_prescan == (0, 1, 0, 1)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, False], [True, False]])).all()

    def test__ones_zeros_full__makes_frame_using_inputs(self):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        ci_frame = ci.MaskedCIFrame.full(
            fill_value=8.0,
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()

        ci_frame = ci.MaskedCIFrame.ones(
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[1.0, 0.0], [1.0, 1.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        ci_frame = ci.MaskedCIFrame.zeros(
            mask=mask,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        assert (ci_frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()

    def test__masked_ci_frame__from_frame__makes_frame_using_inputs(self):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 1, 0, 1)])

        mask = struct.Mask.manual(mask_2d=[[False, True], [False, False]])

        ci_frame = ci.CIFrame.full(
            shape_2d=(2, 2),
            fill_value=8.0,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        ci_frame = ci.MaskedCIFrame.from_ci_frame(ci_frame=ci_frame, mask=mask)

        assert (ci_frame == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert ci_frame.ci_pattern.regions == [(0, 1, 0, 1)]
        assert ci_frame.original_roe_corner == (1, 0)
        assert ci_frame.parallel_overscan == (0, 1, 0, 1)
        assert ci_frame.serial_prescan == (1, 2, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()

        ci_frame = ci.CIFrame.full(
            shape_2d=(2, 2),
            fill_value=8.0,
            ci_pattern=pattern,
            roe_corner=(0, 0),
            parallel_overscan=(0, 1, 0, 1),
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 2, 0, 2),
        )

        ci_frame = ci.MaskedCIFrame.from_ci_frame(ci_frame=ci_frame, mask=mask)

        assert (ci_frame == np.array([[8.0, 0.0], [8.0, 8.0]])).all()
        assert ci_frame.ci_pattern.regions == [(1, 2, 0, 1)]
        assert ci_frame.original_roe_corner == (0, 0)
        assert ci_frame.parallel_overscan == (1, 2, 0, 1)
        assert ci_frame.serial_prescan == (0, 1, 1, 2)
        assert ci_frame.serial_overscan == (0, 2, 0, 2)
        assert (ci_frame.mask == np.array([[False, True], [False, False]])).all()


class TestCiRegionsArray:
    def test__1_ci_region__extracted_correctly(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])
        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        assert (
            ci_frame.ci_regions_frame
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

    def test__2_ci_regions__extracted_correctly(self):
        pattern = ci.CIPatternUniform(
            normalization=10.0, regions=[(0, 1, 1, 2), (2, 3, 1, 3)]
        )
        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        assert (
            ci_frame.ci_regions_frame
            == np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()


class TestNonCIRegionFrame:
    def test__1_ci_region__parallel_overscan_is_entire_image__extracts_everything_between_its_columns(
        self
    ):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(array=arr, ci_pattern=pattern, roe_corner=(1, 0))

        assert (
            ci_frame.non_ci_regions_frame
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
            )
        ).all()
        assert ci_frame.non_ci_regions_frame.ci_pattern.regions == [(0, 3, 0, 3)]

    def test__same_as_above_but_2_ci_regions(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, ci_pattern=pattern)

        assert (
            ci_frame.non_ci_regions_frame
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
        assert ci_frame.non_ci_regions_frame.ci_pattern.regions == [
            (0, 1, 0, 3),
            (3, 4, 0, 3),
        ]


class TestParallelNonCIRegionFrame:
    def test__1_ci_region__parallel_overscan_is_entire_image__extracts_everything_but_removes_serial_scans(
        self
    ):

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            ci_pattern=pattern,
            roe_corner=(1, 0),
            serial_prescan=(3, 4, 2, 3),
            serial_overscan=(3, 4, 0, 1),
        )

        assert (
            ci_frame.parallel_non_ci_regions_frame
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
            )
        ).all()
        assert ci_frame.parallel_non_ci_regions_frame.ci_pattern.regions == [
            (0, 3, 0, 3)
        ]

    def test__same_as_above_but_2_ci_regions(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            ci_pattern=pattern,
            serial_prescan=(1, 2, 0, 3),
            serial_overscan=(0, 1, 0, 1),
        )

        assert (
            ci_frame.parallel_non_ci_regions_frame
            == np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [6.0, 7.0, 8.0],
                    [0.0, 0.0, 0.0],
                    [12.0, 13.0, 14.0],
                ]
            )
        ).all()
        assert ci_frame.parallel_non_ci_regions_frame.ci_pattern.regions == [
            (0, 1, 0, 3),
            (3, 4, 0, 3),
        ]


class TestParallelEdgesAndTrailsFrame:
    def test__front_edge_only__multiple_rows__new_frame_contains_only_edge(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(front_edge_rows=(0, 1))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(front_edge_rows=(0, 2))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]

    def test__trails_only__new_frame_contains_only_trails(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(trails_rows=(0, 1))

        assert (
            new_ci_frame
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(trails_rows=(0, 2))

        assert (
            new_ci_frame
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
        assert new_ci_frame.ci_pattern.regions == [(0, 4, 0, 3)]

    def test__front_edge_and_trails__2_rows_of_each__new_frame_is_edge_and_trail(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 3)])

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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(
            front_edge_rows=(0, 2), trails_rows=(0, 2)
        )

        assert (
            new_ci_frame
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
        assert new_ci_frame.ci_pattern.regions == [(0, 4, 0, 3)]

    def test__front_edge_and_trails__2_regions__1_row_of_each__new_frame_is_edge_and_trail(
        self
    ):
        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.parallel_edges_and_trails_frame(
            front_edge_rows=(0, 1), trails_rows=(0, 1)
        )

        assert (
            new_ci_frame
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
        assert new_ci_frame.ci_pattern.regions == [(0, 1, 0, 3), (3, 4, 0, 3)]


class TestParallelCalibrationFrame:
    def test__columns_0_to_1__extracts_1_column_left_hand_side_of_array(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        extracted_frame = ci_frame.parallel_calibration_frame_from_columns(
            columns=(0, 1)
        )

        assert (extracted_frame == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()
        assert extracted_frame.ci_pattern.regions == [(0, 3, 0, 1)]

    def test__columns_1_to_3__extracts_2_columns_middle_and_right_of_array(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        extracted_frame = ci_frame.parallel_calibration_frame_from_columns(
            columns=(1, 3)
        )

        assert (
            extracted_frame
            == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ).all()
        assert extracted_frame.ci_pattern.regions == [(0, 5, 0, 2)]

    def test__parallel_extracted_mask(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 5, 0, 3)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        mask = ci.CIMask(mask_2d=np.full(fill_value=False, shape=arr.shape))

        mask[0, 1] = True

        extracted_mask = ci_frame.parallel_calibration_mask_from_mask_and_columns(
            mask, columns=(1, 3)
        )

        assert (
            extracted_mask
            == np.array(
                [
                    [True, False],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            )
        ).all()


class TestSerialEdgesAndTrailsFrame:
    def test__front_edge_only__multiple_columns__new_frame_is_only_edge(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )
        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(front_edge_columns=(0, 1))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
            )
        ).all()

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(front_edge_columns=(0, 2))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 3, 0, 3)]

    def test__trails_only__multiple_columns__new_frame_is_only_that_trail(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )
        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(trails_columns=(0, 1))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]]
            )
        ).all()

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(trails_columns=(0, 2))

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 3, 0, 2)]

    def test__front_edge_and_trails__2_columns_of_each__new_frame_is_edge_and_trail(
        self
    ):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 3, 0, 2)])

        arr = np.array(
            [
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ]
        )
        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 2)
        )

        assert (
            new_ci_frame
            == np.array(
                [
                    [0.0, 0.0, 1.1, 2.0, 0.0],
                    [4.0, 0.0, 1.1, 6.0, 0.0],
                    [8.0, 0.0, 1.1, 10.0, 0.0],
                ]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 3, 0, 2)]

    def test__front_edge_and_trails__2_regions_1_column_of_each__new_frame_is_edge_and_trail(
        self
    ):
        pattern = ci.CIPatternUniform(
            normalization=10.0, regions=[(0, 3, 0, 1), (0, 3, 3, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ]
        )
        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        new_ci_frame = ci_frame.serial_edges_and_trails_frame(
            front_edge_columns=(0, 1), trails_columns=(0, 1)
        )

        assert (
            new_ci_frame
            == np.array(
                [
                    [0.0, 1.0, 0.0, 2.0, 3.0],
                    [4.0, 5.0, 0.0, 6.0, 7.0],
                    [8.0, 9.0, 0.0, 10.0, 11.0],
                ]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 3, 0, 1), (0, 3, 3, 4)]


class TestSerialAllTrailsFrame:
    def test__1_ci_region__multiple_serial_trail__extracts_all_trails(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(0, 4, 0, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_overscan=(0, 4, 2, 3),
        )

        new_ci_frame = ci_frame.serial_trails_frame

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 4, 0, 2)]

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_overscan=(0, 4, 2, 4),
        )

        new_ci_frame = ci_frame.serial_trails_frame

        assert (
            new_ci_frame
            == np.array(
                [
                    [0.0, 0.0, 2.0, 0.5],
                    [0.0, 0.0, 5.0, 0.5],
                    [0.0, 0.0, 8.0, 0.5],
                    [0.0, 0.0, 11.0, 0.5],
                ]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 4, 0, 2)]

    def test__2_ci_regions__2_serial_trail__extracts_all_trails(self):
        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_overscan=(0, 4, 2, 4),
        )

        new_ci_frame = ci_frame.serial_trails_frame

        assert (
            new_ci_frame
            == np.array(
                [
                    [0.0, 0.0, 2.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 8.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(0, 1, 0, 2), (2, 3, 0, 2)]


class TestSerialOverScanAboveTrailsFrame:
    def test__1_ci_region__serial_trails_go_over_hand_columns(self):
        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(1, 3, 1, 2)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 2, 4),
        )

        new_ci_frame = ci_frame.serial_overscan_no_trails_frame

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(1, 3, 1, 2)]

        pattern = ci.CIPatternUniform(normalization=10.0, regions=[(1, 3, 1, 3)])

        arr = np.array(
            [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        new_ci_frame = ci_frame.serial_overscan_no_trails_frame

        assert (
            new_ci_frame
            == np.array(
                [[0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            )
        ).all()
        assert new_ci_frame.ci_pattern.regions == [(1, 3, 1, 3)]

    def test__2_ci_regions__serial_trails_go_over_1_right_hand_column__1_pixel_above_each_kept(
        self
    ):
        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 5, 0, 1),
            serial_overscan=(0, 5, 3, 4),
        )

        new_ci_frame = ci_frame.serial_overscan_no_trails_frame

        assert (
            new_ci_frame
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
        assert new_ci_frame.ci_pattern.regions == [(1, 2, 1, 3), (3, 4, 1, 3)]


class TestSerialCalibrationFrame:
    def test__ci_region_across_all_image__column_0__extracts_all_columns(self):

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 5)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
        )

        serial_frame = ci_frame.serial_calibration_frame_from_rows(rows=(0, 3))

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
        assert serial_frame.ci_pattern.regions == [(0, 3, 1, 5)]
        assert serial_frame.parallel_overscan == None
        assert serial_frame.serial_prescan == (0, 3, 0, 1)
        assert serial_frame.serial_overscan == None
        assert serial_frame.pixel_scales == None

    def test__2_ci_regions__both_extracted(self):

        pattern = ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 3), (2, 3, 1, 3)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        serial_frame = ci_frame.serial_calibration_frame_from_rows(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.ci_pattern.regions == [(0, 1, 1, 3), (1, 2, 1, 3)]
        assert serial_frame.parallel_overscan == None
        assert serial_frame.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == None

    def test__rows_cuts_out_bottom_row(self):

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 2, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        serial_frame = ci_frame.serial_calibration_frame_from_rows(rows=(0, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.ci_pattern.regions == [(0, 2, 1, 4)]
        assert serial_frame.parallel_overscan == None
        assert serial_frame.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == None

    def test__extract_two_regions_and_cut_bottom_row_from_each(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 5, 0, 1),
            serial_overscan=(0, 5, 3, 4),
        )

        serial_frame = ci_frame.serial_calibration_frame_from_rows(rows=(0, 1))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 5.0, 5.0, 5.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.ci_pattern.regions == [(0, 1, 1, 4), (1, 2, 1, 4)]
        assert serial_frame.parallel_overscan == None
        assert serial_frame.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == None

    def test__extract_two_regions_and_cut_top_row_from_each(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            roe_corner=(1, 0),
            ci_pattern=pattern,
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        serial_frame = ci_frame.serial_calibration_frame_from_rows(rows=(1, 2))

        assert (
            serial_frame
            == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
        ).all()
        assert serial_frame.original_roe_corner == (1, 0)
        assert serial_frame.ci_pattern.regions == [(0, 1, 1, 4), (1, 2, 1, 4)]
        assert serial_frame.parallel_overscan == None
        assert serial_frame.serial_prescan == (0, 2, 0, 1)
        assert serial_frame.serial_overscan == (0, 2, 3, 4)
        assert serial_frame.pixel_scales == None

    def test__extract_mask_as_above(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, ci_pattern=pattern, roe_corner=(1, 0))

        mask = ci.CIMask(mask_2d=np.full(fill_value=False, shape=arr.shape))

        mask[1, 1] = True
        mask[4, 3] = True

        serial_frame = ci_frame.serial_calibration_mask_from_mask_and_rows(
            mask=mask, rows=(1, 2)
        )

        assert (
            serial_frame
            == np.array(
                [[False, True, False, False, False], [False, False, False, True, False]]
            )
        ).all()


class TestSerialCalibrationArrays:
    def test__different_ci_regions__extracts_all_columns(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 5)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        serial_region = ci_frame.serial_calibration_sub_arrays

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

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 0, 4)])

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        serial_region = ci_frame.serial_calibration_sub_arrays

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

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        serial_region = ci_frame.serial_calibration_sub_arrays

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

    def test__2_ci_regions__both_extracted(self):
        pattern = ci.CIPatternUniform(
            normalization=1.0, regions=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        serial_region = ci_frame.serial_calibration_sub_arrays

        assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
        assert (serial_region[1] == np.array([[0.0, 1.0, 4.0, 4.0, 4.0]])).all()


class TestParallelFrontEdgeArrays:
    def test__1_region__extracts_front_edges_correctly(self):

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(1, 4, 0, 3)])

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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edge = ci_frame.parallel_front_edge_arrays(rows=(0, 1))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0]])).all()

        front_edge = ci_frame.parallel_front_edge_arrays(rows=(1, 2))
        assert (front_edge[0] == np.array([[2.0, 2.0, 2.0]])).all()

        front_edge = ci_frame.parallel_front_edge_arrays(rows=(2, 3))
        assert (front_edge[0] == np.array([[3.0, 3.0, 3.0]])).all()

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(1, 5, 0, 3)])

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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edge = ci_frame.parallel_front_edge_arrays(rows=(0, 2))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

        front_edge = ci_frame.parallel_front_edge_arrays(rows=(1, 4))
        assert (
            front_edge[0]
            == np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

    def test__2_regions__extracts_rows_correctly(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edges = ci_frame.parallel_front_edge_arrays(rows=(0, 1))
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

        front_edges = ci_frame.parallel_front_edge_arrays(rows=(1, 2))
        assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
        assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

        front_edges = ci_frame.parallel_front_edge_arrays(rows=(2, 3))
        assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

        front_edges = ci_frame.parallel_front_edge_arrays(rows=(0, 3))
        assert (
            front_edges[0]
            == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        ).all()
        assert (
            front_edges[1]
            == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        # First fronrt edge arrays:
        #
        # [[1.0, 1.0, 1.0],
        #  [2.0, 2.0, 2.0],
        #  [3.0, 3.0, 3.0]])

        # Second front edge arrays:

        # [[5.0, 5.0, 5.0],
        #  [6.0, 6.0, 6.0],
        #  [7.0, 7.0, 7.0]]

        stacked_front_edges = ci_frame.parallel_front_edge_stacked_array(rows=(0, 3))

        assert (
            stacked_front_edges
            == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        ).all()

        front_edge_line = ci_frame.parallel_front_edge_line_binned_over_columns(
            rows=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    def test__no_rows_specified__uses_smallest_ci_pattern_rows(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edges = ci_frame.parallel_front_edge_arrays()
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])).all()

        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])).all()

        stacked_front_edges = ci_frame.parallel_front_edge_stacked_array()

        assert (
            stacked_front_edges == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

        front_edge_line = ci_frame.parallel_front_edge_line_binned_over_columns()

        assert (front_edge_line == np.array([3.0, 4.0])).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        front_edges = ci_frame.parallel_front_edge_arrays(rows=(0, 3))

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

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
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

        stacked_front_edges = ci_frame.parallel_front_edge_stacked_array(rows=(0, 3))

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

        front_edge_line = ci_frame.parallel_front_edge_line_binned_over_columns(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
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

        stacked_front_edges = ci_frame.parallel_front_edge_stacked_array(rows=(0, 3))

        assert (
            stacked_front_edges.mask
            == np.ma.array(
                [[False, False, True], [False, False, False], [False, False, False]]
            )
        ).all()


class TestParallelTrailsArray:
    def test__1_region__extracts_trails_correctly(self):

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(1, 3, 0, 3)])

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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.parallel_trails_arrays(rows=(0, 1))

        assert (trails == np.array([[3.0, 3.0, 3.0]])).all()
        trails = ci_frame.parallel_trails_arrays(rows=(1, 2))

        assert (trails == np.array([[4.0, 4.0, 4.0]])).all()
        trails = ci_frame.parallel_trails_arrays(rows=(2, 3))
        assert (trails == np.array([[5.0, 5.0, 5.0]])).all()

        trails = ci_frame.parallel_trails_arrays(rows=(0, 2))
        assert (trails == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

        trails = ci_frame.parallel_trails_arrays(rows=(1, 3))
        assert (trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

        trails = ci_frame.parallel_trails_arrays(rows=(1, 4))
        assert (
            trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()

    def test__2_regions__extracts_rows_correctly(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.parallel_trails_arrays(rows=(0, 1))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0]])).all()

        trails = ci_frame.parallel_trails_arrays(rows=(0, 2))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails = ci_frame.parallel_trails_arrays(rows=(1, 4))
        assert (
            trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()
        assert (
            trails[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        trails = ci_frame.parallel_trails_arrays(rows=(0, 2))
        assert (
            trails[0].mask == np.array([[False, True, True], [False, False, False]])
        ).all()

        assert (
            trails[1].mask == np.array([[False, False, False], [True, False, False]])
        ).all()

    def test__stacked_array_and_binned_line__2_regions__no_masking(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        # [3.0, 3.0, 3.0],
        #  [4.0, 4.0, 4.0]]

        # Array 2:

        # [[6.0, 6.0, 6.0],
        # [7.0, 7.0, 7.0]]

        stacked_trails = ci_frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

        trails_line = ci_frame.parallel_trails_line_binned_over_columns(rows=(0, 2))

        assert (trails_line == np.array([4.5, 5.5])).all()

    def test__same_as_above__include_masking(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        # [3.0, 3.0, 3.0],
        #  [4.0, 4.0, 4.0]]

        # Array 2:

        # [[6.0, 6.0, 6.0],
        # [7.0, 7.0, 7.0]]

        stacked_trails = ci_frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (stacked_trails == np.array([[4.5, 6.0, 6.0], [4.0, 5.5, 5.5]])).all()
        assert (
            stacked_trails.mask
            == np.array([[False, False, False], [False, False, False]])
        ).all()

        trails_line = ci_frame.parallel_trails_line_binned_over_columns(rows=(0, 2))

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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        stacked_trails = ci_frame.parallel_trails_stacked_array(rows=(0, 2))

        assert (
            stacked_trails.mask
            == np.array([[False, False, True], [False, False, False]])
        ).all()

    def test__no_rows_specified__uses_smallest_parallel_trails_size(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.parallel_trails_arrays()
        assert (trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()
        assert (trails[1] == np.array([[8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])).all()

        stacked_trails = ci_frame.parallel_trails_stacked_array()

        assert (stacked_trails == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails_line = ci_frame.parallel_trails_line_binned_over_columns()

        assert (trails_line == np.array([6.0, 7.0])).all()


class TestSerialFrontEdgeArrays:
    def test__1_region__extracts_multiple_front_edges_correctly(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #       /| Front Edge

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edge = ci_frame.serial_front_edge_arrays(columns=(0, 1))

        assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

        front_edge = ci_frame.serial_front_edge_arrays(columns=(1, 2))

        assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

        front_edge = ci_frame.serial_front_edge_arrays(columns=(2, 3))

        assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 5)])

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edge = ci_frame.serial_front_edge_arrays(columns=(0, 2))

        assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        front_edge = ci_frame.serial_front_edge_arrays(columns=(1, 4))

        assert (
            front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        ).all()

    def test__2_regions__extracts_columns_correctly(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edges = ci_frame.serial_front_edge_arrays(columns=(0, 1))

        assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edges = ci_frame.serial_front_edge_arrays(columns=(1, 2))

        assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edges = ci_frame.serial_front_edge_arrays(columns=(2, 3))

        assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edges = ci_frame.serial_front_edge_arrays(columns=(0, 3))

        assert (
            front_edges[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edges[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        front_edges = ci_frame.serial_front_edge_arrays(columns=(0, 3))

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

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        stacked_front_edges = ci_frame.serial_front_edge_stacked_array(columns=(0, 3))

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

        front_edge_line = ci_frame.serial_front_edge_line_binned_over_rows(
            columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    def test__same_as_above__include_masking(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        stacked_front_edges = ci_frame.serial_front_edge_stacked_array(columns=(0, 3))

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

        front_edge_line = ci_frame.serial_front_edge_line_binned_over_rows(
            columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 10.0 / 3.0, 5.0])).all()

        mask = np.array(
            [
                [False, False, False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, True, True, False, False],
                [False, False, False, False, False, False, False, True, False, False],
            ]
        )

        #                        /| FE 1                       /| FE 2

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        stacked_front_edges = ci_frame.serial_front_edge_stacked_array(columns=(0, 3))

        assert (
            stacked_front_edges.mask
            == np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            )
        ).all()

    def test__no_columns_specified_so_uses_smallest_charge_injection_region_column_size(
        self
    ):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        front_edges = ci_frame.serial_front_edge_arrays()

        assert (front_edges[0] == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        assert (front_edges[1] == np.array([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]])).all()

        stacked_front_edges = ci_frame.serial_front_edge_stacked_array()

        assert (
            stacked_front_edges == np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
        ).all()

        front_edge_line = ci_frame.serial_front_edge_line_binned_over_rows()

        assert (front_edge_line == np.array([3.0, 4.0])).all()


class TestSerialTrailsArrays:
    def test__1_region__extracts_multiple_trails_correctly(self):
        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 3, 1, 4)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )

        #                                    /| Trails Begin

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.serial_trails_arrays(columns=(0, 1))

        assert (trails == np.array([[4.0], [4.0], [4.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(1, 2))

        assert (trails == np.array([[5.0], [5.0], [5.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(2, 3))

        assert (trails == np.array([[6.0], [6.0], [6.0]])).all()

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.serial_trails_arrays(columns=(0, 2))

        assert (trails == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(1, 4))

        assert (
            trails == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__2_regions__extracts_columns_correctly(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        trails = ci_frame.serial_trails_arrays(columns=(0, 1))

        assert (trails[0] == np.array([[4.0], [4.0], [4.0]])).all()
        assert (trails[1] == np.array([[8.0], [8.0], [8.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(1, 2))

        assert (trails[0] == np.array([[5.0], [5.0], [5.0]])).all()
        assert (trails[1] == np.array([[9.0], [9.0], [9.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(2, 3))

        assert (trails[0] == np.array([[6.0], [6.0], [6.0]])).all()
        assert (trails[1] == np.array([[10.0], [10.0], [10.0]])).all()

        trails = ci_frame.serial_trails_arrays(columns=(0, 3))

        assert (
            trails[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        ).all()

        assert (
            trails[1]
            == np.array([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
        ).all()

    def test__masked_frame__extracted_mask_and_masked_array_are_given(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        trails = ci_frame.serial_trails_arrays(columns=(0, 3))

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

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = ci_frame.serial_trails_stacked_array(columns=(0, 3))

        assert (
            stacked_trails
            == np.array([[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 7.0, 8.0]])
        ).all()

        trails_line = ci_frame.serial_trails_line_binned_over_rows(columns=(0, 3))

        assert (trails_line == np.array([6.0, 7.0, 8.0])).all()

    def test__same_as_above__include_masking(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        # [[4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0],
        #  [4.0, 5.0, 6.0]]

        # [[8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0],
        #  [8.0, 9.0, 10.0]]

        stacked_trails = ci_frame.serial_trails_stacked_array(columns=(0, 3))

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

        trails_line = ci_frame.serial_trails_line_binned_over_rows(columns=(0, 3))

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

        ci_frame = ci.MaskedCIFrame.manual(
            array=arr, mask=mask, roe_corner=(1, 0), ci_pattern=pattern
        )

        stacked_trails = ci_frame.serial_trails_stacked_array(columns=(0, 3))

        assert (
            stacked_trails.mask
            == np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            )
        ).all()

    def test__no_columns_specified_so_uses_full_serial_overscan(self):

        pattern = ci.CIPatternUniform(
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

        ci_frame = ci.CIFrame.manual(
            array=arr,
            ci_pattern=pattern,
            serial_overscan=struct.Region((0, 1, 0, 4)),
            serial_prescan=struct.Region((0, 1, 0, 1)),
            parallel_overscan=struct.Region((0, 1, 0, 1)),
            roe_corner=(1, 0),
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


class TestExtractions:
    def test__parallel_serial_calibration_section__extracts_everything(self):

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        assert (ci_frame.parallel_serial_calibration_frame == arr).all()

        pattern = ci.CIPatternUniform(normalization=1.0, regions=[(0, 1, 0, 1)])

        arr = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )

        ci_frame = ci.CIFrame.manual(array=arr, roe_corner=(1, 0), ci_pattern=pattern)

        assert (ci_frame.parallel_serial_calibration_frame == arr).all()

    def test__smallest_parallel_trails_rows_to_frame_edge__x2_ci_region__bottom_frame_geometry(
        self
    ):

        pattern = ci.CIPatternUniform(
            normalization=10.0, regions=[(0, 3, 0, 3), (5, 7, 0, 3)]
        )

        ci_frame = ci.CIFrame.manual(
            array=np.ones((10, 5)), roe_corner=(1, 0), ci_pattern=pattern
        )

        assert ci_frame.smallest_parallel_trails_rows_to_frame_edge == 2

        ci_frame = ci.CIFrame.manual(
            array=np.ones((8, 5)), roe_corner=(1, 0), ci_pattern=pattern
        )

        assert ci_frame.smallest_parallel_trails_rows_to_frame_edge == 1

    def test__serial_trails_columns__extract_two_columns__second_and_third__takes_coordinates_after_right_of_region(
        self, ci_pattern_7x7
    ):

        ci_frame = ci.CIFrame.manual(
            array=np.ones((10, 10)),
            ci_pattern=ci_pattern_7x7,
            serial_overscan=struct.Region((0, 1, 0, 10)),
            serial_prescan=struct.Region((0, 1, 0, 1)),
            parallel_overscan=struct.Region((0, 1, 0, 1)),
            roe_corner=(1, 0),
        )

        assert ci_frame.serial_trails_columns == 10

        ci_frame = ci.CIFrame.manual(
            array=np.ones((50, 50)),
            ci_pattern=ci_pattern_7x7,
            serial_overscan=struct.Region((0, 1, 0, 50)),
            serial_prescan=struct.Region((0, 1, 0, 1)),
            parallel_overscan=struct.Region((0, 1, 0, 1)),
            roe_corner=(1, 0),
        )

        assert ci_frame.serial_trails_columns == 50

    def test__parallel_trail_size_to_frame_edge__parallel_trail_size_to_edge(self):

        pattern = ci.CIPatternUniform(
            normalization=1.0, regions=[struct.Region(region=(0, 3, 0, 3))]
        )

        ci_frame = ci.CIFrame.manual(
            array=np.ones((5, 100)), ci_pattern=pattern, roe_corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 2

        ci_frame = ci.CIFrame.manual(
            array=np.ones((7, 100)), ci_pattern=pattern, roe_corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 4

        pattern = ci.CIPatternUniform(
            normalization=1.0,
            regions=[
                struct.Region(region=(0, 2, 0, 3)),
                struct.Region(region=(5, 8, 0, 3)),
                struct.Region(region=(11, 14, 0, 3)),
            ],
        )

        ci_frame = ci.CIFrame.manual(
            array=np.ones((15, 100)), ci_pattern=pattern, roe_corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 1

        ci_frame = ci.CIFrame.manual(
            array=np.ones((20, 100)), ci_pattern=pattern, roe_corner=(1, 0)
        )

        assert ci_frame.parallel_trail_size_to_frame_edge == 6
