import numpy as np

from autocti.charge_injection import ci_frame
from autocti.data import mask as msk


class MockPattern(object):

    def __init__(self):
        pass


class TestMaskRemoveRegions:

    def test__remove_one_region(self):

        mask = msk.Mask.from_masked_regions(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                            masked_regions=[(0, 3, 2, 3)])

        assert (mask == np.array([[False, False, True],
                                  [False, False, True],
                                  [False, False, True]])).all()

    def test__remove_two_regions(self):

        mask = msk.Mask.from_masked_regions(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                            masked_regions=[(0, 3, 2, 3), (0, 2, 0, 2)])

        assert (mask == np.array([[True, True, True],
                                  [True, True, True],
                                  [False, False, True]])).all()


class TestCosmicRayMask:

    def test__cosmic_ray_mask_included_in_total_mask(self):

        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image)

        assert (mask == np.array([[False, False, False],
                                  [False, True, False],
                                  [False, False, False]])).all()


class TestMaskCosmicsBottomLeftGeometry:

    def test__mask_one_cosmic_ray_with_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
        cosmic_ray_image = np.array([[False, True, False],
                                [False, False, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2)

        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [False, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [True, False, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2)

        assert (mask == np.array([[False, False, False],
                                  [True, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [False, True, True],
                                  [False, True, True]])).all()

    def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False, False],
                                [False, True, False, False],
                                [False, False, False, False],
                                [False, False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(4, 4), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2)

        assert (mask == np.array([[False, False, False, False],
                                  [False, True, True, True],
                                  [False, True, True, True],
                                  [False, True, True, True]])).all()


class TestMaskCosmicsBottomRightGeometry:

    def test__mask_one_cosmic_ray_with_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
        cosmic_ray_image = np.array([[False, True, False],
                                [False, False, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2)

        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [True, True, False],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, False, True],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2)

        assert (mask == np.array([[False, False, False],
                                  [True, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [True, True, False],
                                  [True, True, False]])).all()

    def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False, False],
                                [False, False, False, True],
                                [False, False, False, False],
                                [False, False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(4, 4), frame_geometry=ci_frame.FrameGeometry.euclid_bottom_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2)

        assert (mask == np.array([[False, False, False, False],
                                  [False, True, True, True],
                                  [False, True, True, True],
                                  [False, True, True, True]])).all()


class TestMaskCosmicsTopLeftGeometry:

    def test__mask_one_cosmic_ray_with_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1)

        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, False, False],
                                [False, True, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2)

        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [False, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [True, False, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2)

        assert (mask == np.array([[False, False, False],
                                  [True, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1)

        assert (mask == np.array([[False, True, True],
                                  [False, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False, False],
                                [False, False, False, False],
                                [False, False, False, False],
                                [False, True, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(4, 4), frame_geometry=ci_frame.FrameGeometry.euclid_top_left(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2)

        assert (mask == np.array([[False, False, False, False],
                                  [False, True, True, True],
                                  [False, True, True, True],
                                  [False, True, True, True]])).all()


class TestMaskCosmicsTopRightGeometry:

    def test__mask_one_cosmic_ray_with_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1)

        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_parallel_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, False, False],
                                [False, True, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2)
        assert (mask == np.array([[False, True, False],
                                  [False, True, False],
                                  [False, True, False]])).all()

    def test__mask_one_cosmic_ray_with_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1)

        assert (mask == np.array([[False, False, False],
                                  [True, True, False],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_longer_serial_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, False, True],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2)

        assert (mask == np.array([[False, False, False],
                                  [True, True, True],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False],
                                [False, True, False],
                                [False, False, False]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(3, 3), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1)

        assert (mask == np.array([[True, True, False],
                                  [True, True, False],
                                  [False, False, False]])).all()

    def test__mask_one_cosmic_ray_with_bigger_diagonal_mask(self):
        cosmic_ray_image = np.array([[False, False, False, False],
                                [False, False, False, False],
                                [False, False, False, False],
                                [False, False, False, True]])

        mask = msk.Mask.from_cosmic_ray_image(shape=(4, 4), frame_geometry=ci_frame.FrameGeometry.euclid_top_right(),
                                              cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2)

        assert (mask == np.array([[False, False, False, False],
                                  [False, True, True, True],
                                  [False, True, True, True],
                                  [False, True, True, True]])).all()
