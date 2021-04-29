import numpy as np
import pytest
import autocti as ac
from autocti.charge_injection.layout_ci import region_list_ci_from
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

        layout = ac.ExtractorParallelFrontEdge(region_list=[(1, 2, 0, 1)])

        assert layout.total_rows_min == 1

        layout = ac.ExtractorParallelFrontEdge(region_list=[(1, 3, 0, 1)])

        assert layout.total_rows_min == 2

        layout = ac.ExtractorParallelFrontEdge(region_list=[(1, 2, 0, 1), (3, 4, 0, 1)])

        assert layout.total_rows_min == 1

        layout = ac.ExtractorParallelFrontEdge(region_list=[(1, 2, 0, 1), (3, 5, 0, 1)])

        assert layout.total_rows_min == 1

    def test__total_columns_minimum(self):

        layout = ac.ExtractorParallelFrontEdge(region_list=[(0, 1, 1, 2)])

        assert layout.total_columns_min == 1

        layout = ac.ExtractorParallelFrontEdge(region_list=[(0, 1, 1, 3)])

        assert layout.total_columns_min == 2

        layout = ac.ExtractorParallelFrontEdge(region_list=[(0, 1, 1, 2), (0, 1, 3, 4)])

        assert layout.total_columns_min == 1

        layout = ac.ExtractorParallelFrontEdge(region_list=[(0, 1, 1, 2), (0, 1, 3, 5)])

        assert layout.total_columns_min == 1


class TestExtractorParallelFrontEdge:
    def test__array_2d_list_from(self, parallel_array, parallel_masked_array):

        extractor = ac.ExtractorParallelFrontEdge(region_list=[(1, 4, 0, 3)])

        front_edge = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0]])).all()

        front_edge = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
        assert (front_edge[0] == np.array([[3.0, 3.0, 3.0]])).all()

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        front_edges = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

        front_edges = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
        assert (front_edges[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (front_edges[1] == np.array([[7.0, 7.0, 7.0]])).all()

        front_edges = extractor.array_2d_list_from(array=parallel_array, rows=(0, 3))
        assert (
            front_edges[0]
            == np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        ).all()
        assert (
            front_edges[1]
            == np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
        ).all()

        front_edges = extractor.array_2d_list_from(
            array=parallel_masked_array, rows=(0, 3)
        )

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

    def test__stacked_array_2d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        stacked_front_edges = extractor.stacked_array_2d_from(
            array=parallel_array, rows=(0, 3)
        )

        assert (
            stacked_front_edges
            == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        ).all()

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        stacked_front_edges = extractor.stacked_array_2d_from(
            array=parallel_array, rows=(0, 2)
        )

        assert (
            stacked_front_edges == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

        stacked_front_edges = extractor.stacked_array_2d_from(
            array=parallel_masked_array, rows=(0, 3)
        )

        assert (
            stacked_front_edges
            == np.ma.array([[3.0, 3.0, 3.0], [4.0, 6.0, 4.0], [3.0, 5.0, 7.0]])
        ).all()
        assert (
            stacked_front_edges.mask
            == np.ma.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

    def test__binned_array_1d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_array, rows=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_array, rows=(0, 2)
        )

        assert (front_edge_line == np.array([3.0, 4.0])).all()

        front_edge_line = extractor.binned_array_1d_from(
            array=parallel_masked_array, rows=(0, 3)
        )

        assert (front_edge_line == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()


class TestExtractorParallelTrails:
    def test__array_2d_list_from(self, parallel_array, parallel_masked_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3)])

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (trails == np.array([[3.0, 3.0, 3.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
        assert (trails == np.array([[5.0, 5.0, 5.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
        assert (trails == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(1, 3))
        assert (trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
        assert (
            trails == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(0, 2))
        assert (trails[0] == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])).all()
        assert (trails[1] == np.array([[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])).all()

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(1, 4))
        assert (
            trails[0] == np.array([[4.0, 4.0, 4.0], [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]])
        ).all()
        assert (
            trails[1] == np.array([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0], [9.0, 9.0, 9.0]])
        ).all()

        trails = extractor.array_2d_list_from(array=parallel_masked_array, rows=(0, 2))

        assert (
            trails[0].mask == np.array([[False, False, True], [False, False, False]])
        ).all()

        assert (
            trails[1].mask == np.array([[False, False, False], [True, False, False]])
        ).all()

    def test__stacked_array_2d_from(self, parallel_array, parallel_masked_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

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

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails_line = extractor.binned_array_1d_from(array=parallel_array, rows=(0, 2))

        assert (trails_line == np.array([4.5, 5.5])).all()

        trails_line = extractor.binned_array_1d_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (trails_line == np.array([5.0, 5.0])).all()


class TestExtractorSerialFrontEdge:
    def test__array_2d_list_from(self, serial_array, serial_masked_array):

        extractor = ac.ExtractorSerialFrontEdge(region_list=[(0, 3, 1, 4)])

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        extractor = ac.ExtractorSerialFrontEdge(region_list=[(0, 3, 1, 5)])

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

        assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        front_edge = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

        assert (
            front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        ).all()

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edges = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edges = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edges = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edges = extractor.array_2d_list_from(array=serial_array, columns=(0, 3))

        assert (
            front_edges[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edges[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

        front_edges = extractor.array_2d_list_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (
            (front_edges[0].mask)
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        assert (
            front_edges[1].mask
            == np.array(
                [[True, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

    def test__stacked_array_2d_from(self, serial_array, serial_masked_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        stacked_front_edges = extractor.stacked_array_2d_from(
            array=serial_array, columns=(0, 3)
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

        stacked_front_edges = extractor.stacked_array_2d_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (
            stacked_front_edges
            == np.array([[1.0, 4.0, 5.0], [3.0, 2.0, 5.0], [3.0, 4.0, 3.0]])
        ).all()
        assert (
            stacked_front_edges.mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__binned_array_1d_from(self, serial_array, serial_masked_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edge_line = extractor.binned_array_1d_from(
            array=serial_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        front_edge_line = extractor.binned_array_1d_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([7.0 / 3.0, 4.0, 13.0 / 3.0])).all()


class TestExtractorSerialTrails:
    def test__array_2d_list_from(self, serial_array, serial_masked_array):

        extractor = ac.ExtractorSerialTrails(region_list=[(0, 3, 1, 4)])

        trails = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (trails == np.array([[4.0], [4.0], [4.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (trails == np.array([[5.0], [5.0], [5.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (trails == np.array([[6.0], [6.0], [6.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(0, 2))

        assert (trails == np.array([[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(1, 4))

        assert (
            trails == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

        extractor = ac.ExtractorSerialTrails(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails = extractor.array_2d_list_from(array=serial_array, columns=(0, 1))

        assert (trails[0] == np.array([[4.0], [4.0], [4.0]])).all()
        assert (trails[1] == np.array([[8.0], [8.0], [8.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(1, 2))

        assert (trails[0] == np.array([[5.0], [5.0], [5.0]])).all()
        assert (trails[1] == np.array([[9.0], [9.0], [9.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(2, 3))

        assert (trails[0] == np.array([[6.0], [6.0], [6.0]])).all()
        assert (trails[1] == np.array([[10.0], [10.0], [10.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_array, columns=(0, 3))

        assert (
            trails[0] == np.array([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])
        ).all()

        assert (trails[1] == np.array([[8.0, 9.0], [8.0, 9.0], [8.0, 9.0]])).all()

        trails = extractor.array_2d_list_from(array=serial_masked_array, columns=(0, 3))

        assert (
            trails[0].mask
            == np.array(
                [[False, True, False], [False, False, True], [False, False, False]]
            )
        ).all()

        assert (
            trails[1].mask == np.array([[False, False], [False, False], [False, False]])
        ).all()

    def test__stacked_array_2d_from(self, serial_array, serial_masked_array):

        extractor = ac.ExtractorSerialTrails(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

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

        extractor = ac.ExtractorSerialTrails(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails_line = extractor.binned_array_1d_from(array=serial_array, columns=(0, 2))

        assert (trails_line == np.array([6.0, 7.0])).all()

        extractor = ac.ExtractorSerialTrails(region_list=[(0, 3, 1, 4), (0, 3, 5, 8)])

        trails_line = extractor.binned_array_1d_from(
            array=serial_masked_array, columns=(0, 2)
        )

        assert (trails_line == np.array([6.0, 23.0 / 3.0])).all()


class TestAbstractLayout2DCI(object):
    def test__check_layout_dimensions__layout_has_more_rows_than_image__1_region(self):

        with pytest.raises(exc.Layout2DCIException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(2, 6), normalization=1.0, region_list=([(0, 3, 0, 1)])
            )

        with pytest.raises(exc.Layout2DCIException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(6, 2), normalization=1.0, region_list=([(0, 1, 0, 3)])
            )

        with pytest.raises(exc.Layout2DCIException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(2, 6),
                normalization=1.0,
                region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]),
            )

        with pytest.raises(exc.Layout2DCIException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(6, 2),
                normalization=1.0,
                region_list=([(0, 3, 0, 1), (0, 1, 0, 3)]),
            )

        with pytest.raises(exc.RegionException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(3, 3), normalization=1.0, region_list=([(-1, 0, 0, 0)])
            )

        with pytest.raises(exc.RegionException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(3, 3), normalization=1.0, region_list=([(0, -1, 0, 0)])
            )

        with pytest.raises(exc.RegionException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(3, 3), normalization=1.0, region_list=([(0, 0, -1, 0)])
            )

        with pytest.raises(exc.RegionException):
            ac.ci.Layout2DCIUniform(
                shape_2d=(3, 3), normalization=1.0, region_list=([(0, 0, 0, -1)])
            )

    def test__rows_between_region_list(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(1, 2, 1, 2)]
        )

        assert layout.rows_between_regions == []

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(1, 2, 1, 2), (3, 4, 3, 4)]
        )

        assert layout.rows_between_regions == [1]

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(1, 2, 1, 2), (4, 5, 3, 4)]
        )

        assert layout.rows_between_regions == [2]

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 10),
            normalization=1.0,
            region_list=[(1, 2, 1, 2), (4, 5, 3, 4), (8, 9, 3, 4)],
        )

        assert layout.rows_between_regions == [2, 3]

    def test__serial_trails_columns(self, layout_ci_7x7):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 10),
            normalization=10.0,
            region_list=[(1, 2, 1, 2)],
            serial_overscan=ac.Region2D((0, 1, 0, 10)),
            serial_prescan=ac.Region2D((0, 1, 0, 1)),
            parallel_overscan=ac.Region2D((0, 1, 0, 1)),
        )

        assert layout.serial_trails_columns == 10

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(50, 50),
            normalization=10.0,
            region_list=[(1, 2, 1, 2)],
            serial_overscan=ac.Region2D((0, 1, 0, 50)),
            serial_prescan=ac.Region2D((0, 1, 0, 1)),
            parallel_overscan=ac.Region2D((0, 1, 0, 1)),
        )

        assert layout.serial_trails_columns == 50

    def test__parallel_trail_size_to_array_edge(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 100),
            normalization=1.0,
            region_list=[ac.Region2D(region=(0, 3, 0, 3))],
        )

        assert layout.parallel_trail_size_to_array_edge == 2

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(7, 100),
            normalization=1.0,
            region_list=[ac.Region2D(region=(0, 3, 0, 3))],
        )

        assert layout.parallel_trail_size_to_array_edge == 4

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(15, 100),
            normalization=1.0,
            region_list=[
                ac.Region2D(region=(0, 2, 0, 3)),
                ac.Region2D(region=(5, 8, 0, 3)),
                ac.Region2D(region=(11, 14, 0, 3)),
            ],
        )

        assert layout.parallel_trail_size_to_array_edge == 1

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(20, 100),
            normalization=1.0,
            region_list=[
                ac.Region2D(region=(0, 2, 0, 3)),
                ac.Region2D(region=(5, 8, 0, 3)),
                ac.Region2D(region=(11, 14, 0, 3)),
            ],
        )

        assert layout.parallel_trail_size_to_array_edge == 6

    def test__with_extracted_regions__region_list_are_extracted_correctly(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 2, 0, 2)]
        )

        layout_extracted = layout.with_extracted_regions(
            extraction_region=ac.Region2D((0, 2, 0, 2))
        )

        assert layout_extracted.region_list == [(0, 2, 0, 2)]

        layout_extracted = layout.with_extracted_regions(
            extraction_region=ac.Region2D((0, 1, 0, 1))
        )

        assert layout_extracted.region_list == [(0, 1, 0, 1)]

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 5),
            normalization=1.0,
            region_list=[(2, 4, 2, 4), (0, 1, 0, 1)],
        )

        layout_extracted = layout.with_extracted_regions(
            extraction_region=ac.Region2D((0, 3, 0, 3))
        )

        assert layout_extracted.region_list == [(2, 3, 2, 3), (0, 1, 0, 1)]

        layout_extracted = layout.with_extracted_regions(
            extraction_region=ac.Region2D((2, 5, 2, 5))
        )

        assert layout_extracted.region_list == [(0, 2, 0, 2)]

        layout_extracted = layout.with_extracted_regions(
            extraction_region=ac.Region2D((8, 9, 8, 9))
        )

        assert layout_extracted.region_list == None

    def test__array_2d_of_regions_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        array_extracted = layout.array_2d_of_regions_from(array=array)

        assert (
            array_extracted
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 1, 2), (2, 3, 1, 3)],
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        array_extracted = layout.array_2d_of_regions_from(array=array)

        assert (
            array_extracted
            == np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

    def test__array_2d_of_non_regions_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        array_extracted = layout.array_2d_of_non_regions_from(array=array)

        assert (
            array_extracted
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            pixel_scales=1.0,
        )

        array_extracted = layout.array_2d_of_non_regions_from(array=array)

        assert (
            array_extracted
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

    def test__array_2d_of_parallel_trails_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 3, 0, 3)],
            serial_prescan=(3, 5, 2, 3),
            serial_overscan=(3, 5, 0, 1),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            pixel_scales=1.0,
        )

        array_extracted = layout.array_2d_of_parallel_trails_from(array=array)

        assert (
            array_extracted
            == np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 13.0, 0.0],
                ]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
            serial_prescan=(1, 2, 0, 3),
            serial_overscan=(0, 1, 0, 1),
        )

        array_extracted = layout.array_2d_of_parallel_trails_from(array=array)

        assert (
            array_extracted.native
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

    def test__array_2d_of_parallel_edges_and_trails_from(self, parallel_array):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3), normalization=10.0, region_list=[(0, 4, 0, 3)]
        )

        new_array = layout.array_2d_of_parallel_edges_and_trails_from(
            array=parallel_array, front_edge_rows=(0, 2), trails_rows=(0, 2)
        )

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [4.0, 4.0, 4.0],
                    [5.0, 5.0, 5.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        )

        new_array = layout.array_2d_of_parallel_edges_and_trails_from(
            array=parallel_array, front_edge_rows=(0, 1), trails_rows=(0, 1)
        )

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__array_2d_for_parallel_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 3, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            pixel_scales=1.0,
        )

        extracted_array = layout.array_2d_for_parallel_calibration_from(
            array=array, columns=(0, 1)
        )

        assert (extracted_array == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 5, 0, 3)]
        )

        extracted_array = layout.array_2d_for_parallel_calibration_from(
            array=array, columns=(1, 3)
        )

        assert (
            extracted_array.native
            == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ).all()

    def test__mask_for_parallel_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 5, 0, 3)]
        )

        mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 3), pixel_scales=1.0)

        mask[0, 1] = True

        extracted_mask = layout.mask_for_parallel_calibration_from(
            mask=mask, columns=(1, 3)
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

    def test__extracted_layout_2d_for_parallel_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 3, 0, 3)]
        )

        extracted_layout = layout.extracted_layout_for_parallel_calibration_from(
            columns=(0, 1)
        )

        assert extracted_layout.region_list == [(0, 3, 0, 1)]

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 5, 0, 3)]
        )

        extracted_layout = layout.extracted_layout_for_parallel_calibration_from(
            columns=(1, 3)
        )

        assert extracted_layout.region_list == [(0, 5, 0, 2)]

    def test__array_2d_of_serial_trails_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(4, 3),
            normalization=10.0,
            region_list=[(0, 4, 0, 2)],
            serial_overscan=(0, 4, 2, 3),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_trails_from(array=array)

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0], [0.0, 0.0, 5.0], [0.0, 0.0, 8.0], [0.0, 0.0, 11.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(4, 4),
            normalization=10.0,
            region_list=[(0, 4, 0, 2)],
            serial_overscan=(0, 4, 2, 4),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_trails_from(array=array)

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

    def test__array_2d_of_serial_overscan_above_trails_from(self):
        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 4),
            normalization=10.0,
            region_list=[(1, 2, 1, 3), (3, 4, 1, 3)],
            serial_prescan=(0, 5, 0, 1),
            serial_overscan=(0, 5, 3, 4),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_overscan_above_trails_from(array=array)

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

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(4, 4),
            normalization=10.0,
            region_list=[(0, 1, 0, 2), (2, 3, 0, 2)],
            serial_overscan=(0, 4, 2, 4),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 0.5],
                [3.0, 4.0, 5.0, 0.5],
                [6.0, 7.0, 8.0, 0.5],
                [9.0, 10.0, 11.0, 0.5],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_overscan_above_trails_from(array=array)

        assert (
            new_array
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.5],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 11.0, 0.5],
                ]
            )
        ).all()

    def test__array_2d_of_serial_edges_and_trails_array(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 4), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_edges_and_trails_array(
            array=array, front_edge_columns=(0, 1)
        )

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]
            )
        ).all()

        new_array = layout.array_2d_of_serial_edges_and_trails_array(
            array=array, front_edge_columns=(0, 2)
        )

        assert (
            new_array
            == np.array(
                [[0.0, 1.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0], [8.0, 9.0, 0.0, 0.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 4), normalization=10.0, region_list=[(0, 3, 0, 2)]
        )

        new_array = layout.array_2d_of_serial_edges_and_trails_array(
            array=array, trails_columns=(0, 1)
        )

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 10.0, 0.0]]
            )
        ).all()

        new_array = layout.array_2d_of_serial_edges_and_trails_array(
            array=array, trails_columns=(0, 2)
        )

        assert (
            new_array
            == np.array(
                [[0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 6.0, 7.0], [0.0, 0.0, 10.0, 11.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=10.0,
            region_list=[(0, 3, 0, 1), (0, 3, 3, 4)],
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 1.1, 2.0, 3.0],
                [4.0, 5.0, 1.1, 6.0, 7.0],
                [8.0, 9.0, 1.1, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_of_serial_edges_and_trails_array(
            array=array, front_edge_columns=(0, 1), trails_columns=(0, 1)
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

    def test__array_2d_list_for_serial_calibration(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5), normalization=1.0, region_list=[(0, 3, 0, 5)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 4.0, 4.0],
            ],
            pixel_scales=1.0,
        )

        serial_region = layout.array_2d_list_for_serial_calibration(array=array)

        assert (
            serial_region[0]
            == np.array(
                [
                    [0.0, 1.0, 2.0, 2.0, 2.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 4.0, 4.0],
                ]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5), normalization=1.0, region_list=[(0, 1, 1, 4), (2, 3, 1, 4)]
        )

        serial_region = layout.array_2d_list_for_serial_calibration(array=array)

        assert (serial_region[0] == np.array([[0.0, 1.0, 2.0, 2.0, 2.0]])).all()
        assert (serial_region[1] == np.array([[0.0, 1.0, 2.0, 4.0, 4.0]])).all()

    def test__array_2d_for_serial_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 3, 1, 5)],
            serial_prescan=(0, 3, 0, 1),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 1.0, 2.0, 3.0, 4.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_for_serial_calibration_from(
            array=array, rows=(0, 3)
        )

        assert (
            new_array.native
            == np.array(
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                ]
            )
        ).all()

        assert new_array.pixel_scales == (1.0, 1.0)

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 2, 1, 4)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        new_array = layout.array_2d_for_serial_calibration_from(
            array=array, rows=(0, 2)
        )

        assert (
            new_array.native
            == np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0]])
        ).all()

        assert new_array.pixel_scales == (1.0, 1.0)

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ],
            pixel_scales=1.0,
        )

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        new_array = layout.array_2d_for_serial_calibration_from(
            array=array, rows=(0, 1)
        )

        assert (
            new_array.native
            == np.array([[0.0, 1.0, 2.0, 2.0, 2.0], [0.0, 1.0, 4.0, 4.0, 4.0]])
        ).all()

        assert new_array.pixel_scales == (1.0, 1.0)

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=1.0,
            region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.array_2d_for_serial_calibration_from(
            array=array, rows=(1, 2)
        )

        assert (
            new_array.native
            == np.array([[0.0, 1.0, 3.0, 3.0, 3.0], [0.0, 1.0, 6.0, 6.0, 6.0]])
        ).all()

        assert new_array.pixel_scales == (1.0, 1.0)

    def test__maks_for_serial_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(0, 2, 1, 4), (3, 5, 1, 4)]
        )

        mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 5), pixel_scales=1.0)

        mask[1, 1] = True
        mask[4, 3] = True

        serial_frame = layout.mask_for_serial_calibration_from(mask=mask, rows=(1, 2))

        assert (
            serial_frame
            == np.array(
                [[False, True, False, False, False], [False, False, False, True, False]]
            )
        ).all()

    def test__extracted_layout_for_serial_calibration_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 3, 1, 5)],
            serial_prescan=(0, 3, 0, 1),
        )

        extracted_layout = layout.extracted_layout_for_serial_calibration_from(
            new_shape_2d=(3, 5), rows=(0, 3)
        )

        assert extracted_layout.original_roe_corner == (1, 0)
        assert extracted_layout.region_list == [(0, 3, 1, 5)]
        assert extracted_layout.parallel_overscan == None
        assert extracted_layout.serial_prescan == (0, 3, 0, 1)
        assert extracted_layout.serial_overscan == None

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 2, 1, 4)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        extracted_layout = layout.extracted_layout_for_serial_calibration_from(
            new_shape_2d=(2, 5), rows=(0, 2)
        )

        assert extracted_layout.original_roe_corner == (1, 0)
        assert extracted_layout.region_list == [(0, 2, 1, 4)]
        assert extracted_layout.parallel_overscan == None
        assert extracted_layout.serial_prescan == (0, 2, 0, 1)
        assert extracted_layout.serial_overscan == (0, 2, 3, 4)

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
            ],
            pixel_scales=1.0,
        )

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(3, 5),
            normalization=1.0,
            region_list=[(0, 1, 1, 3), (2, 3, 1, 3)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        extracted_layout = layout.extracted_layout_for_serial_calibration_from(
            new_shape_2d=(2, 5), rows=(0, 1)
        )

        assert extracted_layout.original_roe_corner == (1, 0)
        assert extracted_layout.region_list == [(0, 1, 1, 3), (1, 2, 1, 3)]
        assert extracted_layout.parallel_overscan == None
        assert extracted_layout.serial_prescan == (0, 2, 0, 1)
        assert extracted_layout.serial_overscan == (0, 2, 3, 4)

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=1.0,
            region_list=[(0, 2, 1, 4), (3, 5, 1, 4)],
            serial_prescan=(0, 3, 0, 1),
            serial_overscan=(0, 3, 3, 4),
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 1.0, 3.0, 3.0, 3.0],
                [0.0, 1.0, 4.0, 4.0, 4.0],
                [0.0, 1.0, 5.0, 5.0, 5.0],
                [0.0, 1.0, 6.0, 6.0, 6.0],
            ],
            pixel_scales=1.0,
        )

        extracted_layout = layout.extracted_layout_for_serial_calibration_from(
            new_shape_2d=(2, 5), rows=(1, 2)
        )

        assert extracted_layout.original_roe_corner == (1, 0)
        assert extracted_layout.region_list == [(0, 1, 1, 4), (1, 2, 1, 4)]
        assert extracted_layout.parallel_overscan == None
        assert extracted_layout.serial_prescan == (0, 2, 0, 1)
        assert extracted_layout.serial_overscan == (0, 2, 3, 4)

    def test__smallest_parallel_trails_rows_to_frame_edge(self,):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 5),
            normalization=10.0,
            region_list=[(0, 3, 0, 3), (5, 7, 0, 3)],
        )

        assert layout.smallest_parallel_trails_rows_to_array_edge == 2

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(8, 5),
            normalization=10.0,
            region_list=[(0, 3, 0, 3), (5, 7, 0, 3)],
        )

        assert layout.smallest_parallel_trails_rows_to_array_edge == 1


class TestLayout2DCIUniform(object):
    def test__pre_cti_ci_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 2, 0, 2)]
        )

        pre_cti_ci = layout.pre_cti_ci_from(shape_native=(3, 3), pixel_scales=1.0)

        assert (
            pre_cti_ci.native
            == np.array([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=20.0,
            region_list=[(0, 2, 0, 2), (2, 3, 2, 3)],
        )
        pre_cti_ci = layout.pre_cti_ci_from(shape_native=(3, 3), pixel_scales=1.0)

        assert (
            pre_cti_ci.native
            == np.array([[20.0, 20.0, 0.0], [20.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=30.0,
            region_list=[(0, 3, 0, 2), (2, 3, 2, 3)],
        )
        pre_cti_ci = layout.pre_cti_ci_from(shape_native=(4, 3), pixel_scales=1.0)

        assert (
            pre_cti_ci.native
            == np.array(
                [
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 30.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()


class TestLayout2DCINonUniform(object):
    def test__region_ci_from(self,):

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=0.0,
        )

        region = layout.region_ci_from(region_dimensions=(3, 3), ci_seed=1)

        assert (
            region
            == np.array(
                [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        region = layout.region_ci_from(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [101.6, 99.4, 99.5], [101.6, 99.4, 99.5]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        region = layout.region_ci_from(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[100.0, 100.0, 100.0], [99.3, 99.3, 99.3], [98.9, 98.9, 98.9]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        region = layout.region_ci_from(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [100.9, 98.7, 98.8], [100.5, 98.3, 98.4]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=100.0,
        )

        region = layout.region_ci_from(region_dimensions=(10, 10), ci_seed=1)

        assert (region > 0).all()

    def test__pre_cti_ci_from__compare_uniform_to_non_uniform(self,):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=10.0, region_list=[(2, 4, 0, 5)]
        )
        pre_cti_ci_0 = layout.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=10.0,
            region_list=[(2, 4, 0, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        pre_cti_ci_1 = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0
        )

        assert (pre_cti_ci_0 == pre_cti_ci_1).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(0, 2, 0, 2), (2, 3, 0, 5)],
        )
        pre_cti_ci_0 = layout.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(0, 2, 0, 2), (2, 3, 0, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        pre_cti_ci_1 = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0
        )

        assert (pre_cti_ci_0 == pre_cti_ci_1).all()

    def test__pre_cti_ci_from__non_uniformity_in_columns(self,):
        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 3, 0, 3)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image[:] = np.round(image[:], 1)

        assert (
            image.native
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

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(1, 4, 1, 3), (1, 4, 4, 5)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image[:] = np.round(image[:], 1)

        assert (
            image.native
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

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(0, 5, 0, 5)],
            row_slope=0.0,
            column_sigma=100.0,
            maximum_normalization=100.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        # Checked ci_seed to ensure the max is above 100.0 without a maximum_normalization
        assert np.max(image) < 100.0

    def test__pre_cti_ci_from__non_uniformity_in_rows(self,):
        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(0, 3, 0, 3)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
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

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image[:] = np.round(image[:], 1)

        assert (
            image.native
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

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image[:] = np.round(image[:], 1)

        assert (
            image.native
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

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 5),
            normalization=100.0,
            region_list=[(0, 2, 0, 5), (3, 5, 0, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image[:] = np.round(image[:], 1)

        assert (image.native[0:2, 0:5] == image.native[3:5, 0:5]).all()


class TestRegionCIFrom:
    def test__region_list_ci_from(self):

        region_list_ci = region_list_ci_from(
            injection_on=10,
            injection_off=10,
            injection_total=1,
            parallel_size=10,
            serial_prescan_size=1,
            serial_size=10,
            serial_overscan_size=1,
            roe_corner=(1, 0),
        )

        assert region_list_ci == [(0, 10, 1, 9)]

        region_list_ci = region_list_ci_from(
            injection_on=10,
            injection_off=10,
            injection_total=2,
            parallel_size=30,
            serial_prescan_size=2,
            serial_size=11,
            serial_overscan_size=4,
            roe_corner=(1, 0),
        )

        assert region_list_ci == [(0, 10, 2, 7), (20, 30, 2, 7)]

        region_list_ci = region_list_ci_from(
            injection_on=5,
            injection_off=10,
            injection_total=3,
            parallel_size=35,
            serial_prescan_size=2,
            serial_size=11,
            serial_overscan_size=4,
            roe_corner=(1, 0),
        )

        assert region_list_ci == [(0, 5, 2, 7), (15, 20, 2, 7), (30, 35, 2, 7)]

        region_list_ci = region_list_ci_from(
            injection_on=200,
            injection_off=200,
            injection_total=5,
            parallel_size=2000,
            serial_prescan_size=51,
            serial_size=2128,
            serial_overscan_size=29,
            roe_corner=(1, 0),
        )

        assert region_list_ci == [
            (0, 200, 51, 2099),
            (400, 600, 51, 2099),
            (800, 1000, 51, 2099),
            (1200, 1400, 51, 2099),
            (1600, 1800, 51, 2099),
        ]
