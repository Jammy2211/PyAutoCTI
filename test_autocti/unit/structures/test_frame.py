import os

import numpy as np
import autocti as ac


path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))


class TestFrameAPI:
    def test__manual__makes_frame_using_inputs__include_rotations(self):

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[1.0, 2.0], [3.0, 4.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()
        assert (frame.original_orientation == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[3.0, 4.0], [1.0, 2.0]])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.scans.parallel_overscan == (1, 2, 0, 1)
        assert frame.scans.serial_prescan == (0, 1, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()
        assert (frame.original_orientation == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(1, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[2.0, 1.0], [4.0, 3.0]])).all()
        assert frame.original_roe_corner == (1, 1)
        assert frame.scans.parallel_overscan == (0, 1, 1, 2)
        assert frame.scans.serial_prescan == (1, 2, 0, 1)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()
        assert (frame.original_orientation == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(0, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[4.0, 3.0], [2.0, 1.0]])).all()
        assert frame.original_roe_corner == (0, 1)
        assert frame.scans.parallel_overscan == (1, 2, 1, 2)
        assert frame.scans.serial_prescan == (0, 1, 0, 1)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()
        assert (frame.original_orientation == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    def test__full_ones_zeros__makes_frame_using_inputs(self):

        frame = ac.Frame.full(
            fill_value=8.0,
            shape_2d=(2, 2),
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[8.0, 8.0], [8.0, 8.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()

        frame = ac.Frame.ones(
            shape_2d=(2, 2),
            roe_corner=(1, 0),
            pixel_scales=1.0,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[1.0, 1.0], [1.0, 1.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()

        frame = ac.Frame.zeros(
            shape_2d=(2, 2),
            roe_corner=(1, 0),
            pixel_scales=1.0,
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[0.0, 0.0], [0.0, 0.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()

    def test__extracted_frame_from_frame_and_extraction_region(self):

        frame = ac.Frame.manual(
            array=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=None,
                serial_prescan=(0, 2, 0, 2),
                serial_overscan=(1, 2, 1, 2),
            ),
        )

        frame = ac.Frame.extracted_frame_from_frame_and_extraction_region(
            frame=frame, extraction_region=ac.Region(region=(1, 3, 1, 3))
        )

        assert (frame == np.array([[5.0, 6.0], [8.0, 9.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == None
        assert frame.scans.serial_prescan == (0, 1, 0, 1)
        assert frame.scans.serial_overscan == (0, 1, 0, 1)
        assert (frame.mask == np.array([[False, False], [False, False]])).all()

    def test__manual_mask__makes_frame_using_inputs__includes_rotation_which_includes_mask(
        self
    ):

        mask = ac.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        frame = ac.Frame.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        frame = ac.Frame.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[3.0, 4.0], [1.0, 0.0]])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.scans.parallel_overscan == (1, 2, 0, 1)
        assert frame.scans.serial_prescan == (0, 1, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [False, True]])).all()

        frame = ac.Frame.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(1, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[0.0, 1.0], [4.0, 3.0]])).all()
        assert frame.original_roe_corner == (1, 1)
        assert frame.scans.parallel_overscan == (0, 1, 1, 2)
        assert frame.scans.serial_prescan == (1, 2, 0, 1)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[True, False], [False, False]])).all()

        frame = ac.Frame.manual_mask(
            array=[[1.0, 2.0], [3.0, 4.0]],
            mask=mask,
            roe_corner=(0, 1),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        assert (frame == np.array([[4.0, 3.0], [0.0, 1.0]])).all()
        assert frame.original_roe_corner == (0, 1)
        assert frame.scans.parallel_overscan == (1, 2, 1, 2)
        assert frame.scans.serial_prescan == (0, 1, 0, 1)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, False], [True, False]])).all()

    def test__from_frame__no_rotation_as_frame_is_correct_orientation(self):

        mask = ac.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(1, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        frame = ac.Frame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[1.0, 0.0], [3.0, 4.0]])).all()
        assert frame.original_roe_corner == (1, 0)
        assert frame.scans.parallel_overscan == (0, 1, 0, 1)
        assert frame.scans.serial_prescan == (1, 2, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()

        mask = ac.Mask2D.manual(mask=[[False, True], [False, False]], pixel_scales=1.0)

        frame = ac.Frame.manual(
            array=[[1.0, 2.0], [3.0, 4.0]],
            pixel_scales=1.0,
            roe_corner=(0, 0),
            scans=ac.Scans(
                parallel_overscan=(0, 1, 0, 1),
                serial_prescan=(1, 2, 1, 2),
                serial_overscan=(0, 2, 0, 2),
            ),
        )

        frame = ac.Frame.from_frame(frame=frame, mask=mask)

        assert (frame == np.array([[3.0, 0.0], [1.0, 2.0]])).all()
        assert frame.original_roe_corner == (0, 0)
        assert frame.scans.parallel_overscan == (1, 2, 0, 1)
        assert frame.scans.serial_prescan == (0, 1, 1, 2)
        assert frame.scans.serial_overscan == (0, 2, 0, 2)
        assert (frame.mask == np.array([[False, True], [False, False]])).all()
