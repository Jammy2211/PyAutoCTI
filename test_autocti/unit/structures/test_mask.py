import numpy as np

import autocti as ac


class MockPattern(object):
    def __init__(self):
        pass


class TestMaskRemoveRegions:
    def test__remove_one_region(self):

        mask = ac.Mask.from_masked_regions(
            shape_2d=(3, 3), masked_regions=[(0, 3, 2, 3)]
        )

        assert (
            mask
            == np.array(
                [[False, False, True], [False, False, True], [False, False, True]]
            )
        ).all()

    def test__remove_two_regions(self):

        mask = ac.Mask.from_masked_regions(
            shape_2d=(3, 3), masked_regions=[(0, 3, 2, 3), (0, 2, 0, 2)]
        )

        assert (
            mask
            == np.array([[True, True, True], [True, True, True], [False, False, True]])
        ).all()


class TestCosmicRayMask:
    def test__cosmic_ray_mask_included_in_total_mask(self):

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(cosmic_ray_image=cosmic_ray_image)

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()


class TestMaskCosmics:
    def test__top_left__mask_cosmic_ray_in_different_directions(self):

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, True, False], [False, False, False], [False, False, False]]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [True, False, False], [False, False, False]]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, True], [False, True, True]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ),
            corner=(0, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False],
                    [False, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            )
        ).all()

    def test__top_right__mask_cosmic_ray_in_different_directions(self):
        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, True, False], [False, False, False], [False, False, False]]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, False], [True, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [
                    [False, False, False, False],
                    [False, False, False, True],
                    [False, False, False, False],
                    [False, False, False, False],
                ]
            ),
            corner=(0, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False],
                    [False, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            )
        ).all()

    def test__bottom_left__mask_cosmic_ray_in_different_directions(self):

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [False, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [True, False, False], [False, False, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, True, True], [False, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, True, False, False],
                ]
            ),
            corner=(1, 0),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False],
                    [False, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            )
        ).all()

    def test__bottom_right__mask_cosmic_ray_in_different_directions(self):

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, False, False], [False, True, False]]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_parallel_buffer=2
        )
        assert (
            mask
            == np.array(
                [[False, True, False], [False, True, False], [False, True, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=1
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, False, True], [False, False, False]]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_serial_buffer=2
        )

        assert (
            mask
            == np.array(
                [[False, False, False], [True, True, True], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=1
        )

        assert (
            mask
            == np.array(
                [[True, True, False], [True, True, False], [False, False, False]]
            )
        ).all()

        cosmic_ray_image = ac.frame(
            array=np.array(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, False, False, True],
                ]
            ),
            corner=(1, 1),
        )

        mask = ac.Mask.from_cosmic_ray_image(
            cosmic_ray_image=cosmic_ray_image, cosmic_ray_diagonal_buffer=2
        )

        assert (
            mask
            == np.array(
                [
                    [False, False, False, False],
                    [False, True, True, True],
                    [False, True, True, True],
                    [False, True, True, True],
                ]
            )
        ).all()
