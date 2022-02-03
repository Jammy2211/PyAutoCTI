import numpy as np
import pytest
import autocti as ac
from autocti.charge_injection.layout import region_list_ci_from
from autocti import exc


@pytest.fixture(name="parallel_array")
def make_parallel_array():
    return ac.Array2D.manual(
        array=[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # <- Front edge .
            [2.0, 2.0, 2.0],  # <- Next front edge row.
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
        ],
        pixel_scales=1.0,
    )


@pytest.fixture(name="parallel_masked_array")
def make_parallel_masked_array(parallel_array):

    mask = ac.Mask2D.manual(
        mask=[
            [False, False, False],
            [False, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, False, False],
            [False, False, False],
        ],
        pixel_scales=1.0,
    )

    return ac.Array2D.manual_mask(array=parallel_array.native, mask=mask)


@pytest.fixture(name="serial_array")
def make_serial_array():
    return ac.Array2D.manual(
        array=[
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ],
        pixel_scales=1.0,
    )


@pytest.fixture(name="serial_masked_array")
def make_serial_masked_array(serial_array):

    mask = ac.Mask2D.manual(
        mask=[
            [False, False, False, False, False, True, False, False, False, False],
            [False, False, True, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, True, False, False],
        ],
        pixel_scales=1.0,
    )

    return ac.Array2D.manual_mask(array=serial_array.native, mask=mask)


class TestAbstractExtractor:
    def test__total_rows_minimum(self):

        layout = ac.Extractor2DParallelFPR(region_list=[(1, 2, 0, 1)])

        assert layout.total_rows_min == 1

        layout = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 1)])

        assert layout.total_rows_min == 2

        layout = ac.Extractor2DParallelFPR(region_list=[(1, 2, 0, 1), (3, 4, 0, 1)])

        assert layout.total_rows_min == 1

        layout = ac.Extractor2DParallelFPR(region_list=[(1, 2, 0, 1), (3, 5, 0, 1)])

        assert layout.total_rows_min == 1

    def test__total_columns_minimum(self):

        layout = ac.Extractor2DParallelFPR(region_list=[(0, 1, 1, 2)])

        assert layout.total_columns_min == 1

        layout = ac.Extractor2DParallelFPR(region_list=[(0, 1, 1, 3)])

        assert layout.total_columns_min == 2

        layout = ac.Extractor2DParallelFPR(region_list=[(0, 1, 1, 2), (0, 1, 3, 4)])

        assert layout.total_columns_min == 1

        layout = ac.Extractor2DParallelFPR(region_list=[(0, 1, 1, 2), (0, 1, 3, 5)])

        assert layout.total_columns_min == 1


class TestExtractorParallelFPR:
    def test__array_2d_list_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3)])

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_array, pixels=(0, 1)
        )
        assert (front_edge_list[0] == np.array([[1.0, 1.0, 1.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_array, pixels=(2, 3)
        )
        assert (front_edge_list[0] == np.array([[3.0, 3.0, 3.0]])).all()

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_array, pixels=(0, 1)
        )
        assert (front_edge_list[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (front_edge_list[1] == np.array([[5.0, 5.0, 5.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_array, pixels=(2, 3)
        )
        assert (front_edge_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (front_edge_list[1] == np.array([[7.0, 7.0, 7.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_array, pixels=(0, 3)
        )
        assert (
            front_edge_list[0]
            == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        ).all()
        assert (
            front_edge_list[1]
            == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
        ).all()

        front_edge_list = extractor.array_2d_list_from(
            array=parallel_masked_array, pixels=(0, 3)
        )

        assert (
            front_edge_list[0].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

        assert (
            front_edge_list[1].mask
            == np.array(
                [[False, False, False], [False, False, False], [True, False, False]]
            )
        ).all()

    def test__stacked_array_2d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 4, 0, 3), (5, 8, 0, 3)])

        stacked_front_edge_list = extractor.stacked_array_2d_from(
            array=parallel_array, pixels=(0, 3)
        )

        assert (
            stacked_front_edge_list
            == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        ).all()

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

        stacked_front_edge_list = extractor.stacked_array_2d_from(
            array=parallel_array, pixels=(0, 2)
        )

        assert (
            stacked_front_edge_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

        stacked_front_edge_list = extractor.stacked_array_2d_from(
            array=parallel_masked_array, pixels=(0, 3)
        )

        assert (
            stacked_front_edge_list
            == np.ma.array([[3.0, 3.0, 3.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
        ).all()
        assert (
            stacked_front_edge_list.mask
            == np.ma.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

    def test__binned_array_1d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_array, pixels=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        extractor = ac.Extractor2DParallelFPR(region_list=[(1, 3, 0, 3), (5, 8, 0, 3)])

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_array, pixels=(0, 2)
        )

        assert (front_edge_line == np.array([3.0, 4.0])).all()

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_masked_array, pixels=(0, 3)
        )

        assert (front_edge_line == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()


class TestExtractorParallelEPER:
    def test__array_2d_list_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3)])

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (trails_list == np.array([[3.0, 3.0, 3.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
        assert (trails_list == np.array([[5.0, 5.0, 5.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
        assert (trails_list == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 3))
        assert (trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
        assert (
            trails_list == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()

        extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (trails_list[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (trails_list[1] == np.array([[6.0, 6.0, 6.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
        assert (trails_list[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
        assert (trails_list[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails_list = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
        assert (
            trails_list[0]
            == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()
        assert (
            trails_list[1]
            == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
        ).all()

        trails_list = extractor.array_2d_list_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (
            trails_list[0].mask
            == np.array([[False, False, True], [False, False, False]])
        ).all()

        assert (
            trails_list[1].mask
            == np.array([[False, False, False], [True, False, False]])
        ).all()

    def test__stacked_array_2d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        stacked_trails = extractor.stacked_array_2d_from(
            array=parallel_array, rows=(0, 2)
        )

        assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

        stacked_trails = extractor.stacked_array_2d_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (stacked_trails == np.array([[4.5, 4.5, 6.0], [4.0, 5.5, 5.5]])).all()
        assert (
            stacked_trails.mask
            == np.array([[False, False, False], [False, False, False]])
        ).all()

    def test__binned_array_1d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.Extractor2DParallelEPER(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails_line = extractor.binned_array_1d_from(array=parallel_array, rows=(0, 2))

        assert (trails_line == np.array([4.5, 5.5])).all()

        trails_line = extractor.binned_array_1d_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (trails_line == np.array([5.0, 5.0])).all()


class TestExtractorSerialFPR:
    def test__array_2d_list_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4)])

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 5)])

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

        assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

        assert (
            front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        ).all()

        extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        front_edge_list = extractor.array_2d_list_from(
            array=serial_array, columns=(0, 1)
        )

        assert (front_edge_list[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edge_list[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=serial_array, columns=(1, 2)
        )

        assert (front_edge_list[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edge_list[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=serial_array, columns=(2, 3)
        )

        assert (front_edge_list[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edge_list[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edge_list = extractor.array_2d_list_from(
            array=serial_array, columns=(0, 3)
        )

        assert (
            front_edge_list[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edge_list[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

        front_edge_list = extractor.array_2d_list_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (
            (front_edge_list[0].mask)
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        assert (
            front_edge_list[1].mask
            == np.array(
                [[True, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

    def test__stacked_array_2d_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        stacked_front_edge_list = extractor.stacked_array_2d_from(
            array=serial_array, columns=(0, 3)
        )

        # [[1.0, 2.0, 3.0],
        #  [1.0, 2.0, 3.0],
        #  [1.0, 2.0, 3.0]]

        # [[5.0, 6.0, 7.0],
        #  [5.0, 6.0, 7.0],
        #  [5.0, 6.0, 7.0]]

        assert (
            stacked_front_edge_list
            == np.array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])
        ).all()

        stacked_front_edge_list = extractor.stacked_array_2d_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (
            stacked_front_edge_list
            == np.array([[1.0, 4.0, 5.0], [3.0, 2.0, 5.0], [3.0, 4.0, 3.0]])
        ).all()
        assert (
            stacked_front_edge_list.mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__binned_array_1d_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialFPR(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        front_edge_line = extractor.binned_array_1d_from(
            array=serial_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        front_edge_line = extractor.binned_array_1d_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([7.0 / 3.0, 4.0, 13.0 / 3.0])).all()


class TestExtractorSerialEPER:
    def test__array_2d_list_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4)])

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (trails_list == np.array([[4.0], [4.0], [4.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (trails_list == np.array([[5.0], [5.0], [5.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (trails_list == np.array([[6.0], [6.0], [6.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

        assert (trails_list == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

        assert (
            trails_list == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

        extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (trails_list[0] == np.array([[4.0], [4.0], [4.0]])).all()
        assert (trails_list[1] == np.array([[8.0], [8.0], [8.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (trails_list[0] == np.array([[5.0], [5.0], [5.0]])).all()
        assert (trails_list[1] == np.array([[9.0], [9.0], [9.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (trails_list[0] == np.array([[6.0], [6.0], [6.0]])).all()
        assert (trails_list[1] == np.array([[10.0], [10.0], [10.0]])).all()

        trails_list = extractor.array_2d_list_from(array=serial_array, columns=(0, 3))

        assert (
            trails_list[0]
            == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        ).all()

        assert (trails_list[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

        trails_list = extractor.array_2d_list_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (
            trails_list[0].mask
            == np.array(
                [[False, True, False], [False, False, True], [False, False, False]]
            )
        ).all()

        assert (
            trails_list[1].mask
            == np.array([[False, False], [False, False], [False, False]])
        ).all()

    def test__stacked_array_2d_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        stacked_trails = extractor.stacked_array_2d_from(
            array=serial_array, columns=(0, 2)
        )

        assert (stacked_trails == np.array([[6.0, 7.0], [6.0, 7.0], [6.0, 7.0]])).all()

        stacked_trails = extractor.stacked_array_2d_from(
            array=serial_masked_array, columns=(0, 2)
        )

        assert (stacked_trails == np.array([[6.0, 9.0], [6.0, 7.0], [6.0, 7.0]])).all()
        assert (
            stacked_trails.mask
            == np.array([[False, False], [False, False], [False, False]])
        ).all()

    def test__binned_array_1d_from(self, serial_array, serial_masked_array):

        extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails_line = extractor.binned_array_1d_from(array=serial_array, columns=(0, 2))

        assert (trails_line == np.array([6.0, 7.0])).all()

        extractor = ac.Extractor2DSerialEPER(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails_line = extractor.binned_array_1d_from(
            array=serial_masked_array, columns=(0, 2)
        )

        assert (trails_line == np.array([6.0, 23.0 / 3.0])).all()
