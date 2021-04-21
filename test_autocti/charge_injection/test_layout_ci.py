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
            [False, False, False, False, False, False, False, False, False, False],
            [False, False, True, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, True, False, False],
        ],
        pixel_scales=1.0,
    )

    return ac.Array2D.manual_mask(array=serial_array.native, mask=mask)


class TestExtractorParallelFrontEdge:
    def test__array_2d_list_from(self, parallel_array):

        extractor = ac.ExtractorParallelFrontEdge(region_list=[(1, 4, 0, 3)])

        front_edge = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (front_edge[0] == np.array([[1.0, 1.0, 1.0]])).all()

        front_edge = extractor.array_2d_list_from(array=parallel_array, rows=(1, 2))
        assert (front_edge[0] == np.array([[2.0, 2.0, 2.0]])).all()

        front_edge = extractor.array_2d_list_from(array=parallel_array, rows=(2, 3))
        assert (front_edge[0] == np.array([[3.0, 3.0, 3.0]])).all()

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        front_edges = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))
        assert (front_edges[0] == np.array([[1.0, 1.0, 1.0]])).all()
        assert (front_edges[1] == np.array([[5.0, 5.0, 5.0]])).all()

        front_edges = extractor.array_2d_list_from(array=parallel_array, rows=(1, 2))
        assert (front_edges[0] == np.array([[2.0, 2.0, 2.0]])).all()
        assert (front_edges[1] == np.array([[6.0, 6.0, 6.0]])).all()

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

    def test__array_2d_list_from__include_mask(self, parallel_masked_array):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

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

    def test__stacked_array_2d_from(self, parallel_array):

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

        stacked_front_edges = extractor.stacked_array_2d_from(array=parallel_array)

        assert (
            stacked_front_edges == np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
        ).all()

    def test__stacked_array_2d_from__include_mask(self, parallel_masked_array):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        # First front edge arrays:
        #
        # [[1.0, 1.0, 1.0],
        #  [2.0, 2.0, 2.0],
        #  [3.0, 3.0, 3.0]])

        # Second front edge arrays:

        # [[5.0, 5.0, 5.0],
        #  [6.0, 6.0, 6.0],
        #  [7.0, 7.0, 7.0]]

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

    def test__binned_over_columns_array_1d_from_columns_from(self, parallel_array):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        front_edge_line = extractor.binned_over_columns_array_1d_from(
            array=parallel_array, rows=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 3, 0, 3), (5, 8, 0, 3)]
        )

        front_edge_line = extractor.binned_over_columns_array_1d_from(
            array=parallel_array
        )

        assert (front_edge_line == np.array([3.0, 4.0])).all()

    def test__binned_over_columns_array_1d_from__include_mask(
        self, parallel_masked_array
    ):

        extractor = ac.ExtractorParallelFrontEdge(
            region_list=[(1, 4, 0, 3), (5, 8, 0, 3)]
        )

        front_edge_line = extractor.binned_over_columns_array_1d_from(
            array=parallel_masked_array, rows=(0, 3)
        )

        assert (front_edge_line == np.array([9.0 / 3.0, 14.0 / 3.0, 5.0])).all()


class TestExtractorParallelTrails:
    def test__array_2d_list_from(self, parallel_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3)])

        trails = extractor.array_2d_list_from(array=parallel_array, rows=(0, 1))

        assert (trails == np.array([[3.0, 3.0, 3.0]])).all()
        trails = extractor.array_2d_list_from(array=parallel_array, rows=(1, 2))

        assert (trails == np.array([[4.0, 4.0, 4.0]])).all()
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

    def test__array_2d_list_from__include_masking(self, parallel_masked_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails = extractor.array_2d_list_from(array=parallel_masked_array, rows=(0, 2))

        assert (
            trails[0].mask == np.array([[False, False, True], [False, False, False]])
        ).all()

        assert (
            trails[1].mask == np.array([[False, False, False], [True, False, False]])
        ).all()

    def test__stacked_array_2d_from(self, parallel_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        stacked_trails = extractor.stacked_array_2d_from(
            array=parallel_array, rows=(0, 2)
        )

        assert (stacked_trails == np.array([[4.5, 4.5, 4.5], [5.5, 5.5, 5.5]])).all()

    def test__stacked_array_2d_from__include_masking(self, parallel_masked_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        stacked_trails = extractor.stacked_array_2d_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (stacked_trails == np.array([[4.5, 4.5, 6.0], [4.0, 5.5, 5.5]])).all()
        assert (
            stacked_trails.mask
            == np.array([[False, False, False], [False, False, False]])
        ).all()

    def test__binned_over_columns_array_1d_from_columns_from(self, parallel_array):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails_line = extractor.binned_over_columns_array_1d_from(
            array=parallel_array, rows=(0, 2)
        )

        assert (trails_line == np.array([4.5, 5.5])).all()

    def test__binned_over_columns_array_1d_from_columns__include_masking(
        self, parallel_masked_array
    ):

        extractor = ac.ExtractorParallelTrails(region_list=[(1, 3, 0, 3), (4, 6, 0, 3)])

        trails_line = extractor.binned_over_columns_array_1d_from(
            array=parallel_masked_array, rows=(0, 2)
        )

        assert (trails_line == np.array([5.0, 5.0])).all()


class TestExtractorSerialFrontEdge:
    def test__array_2d_list_from(self, serial_array):

        extractor = ac.ExtractorSerialFrontEdge(region_list=[(0, 3, 1, 4)])

        front_edge = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(0, 1)
        )

        assert (front_edge == np.array([[1.0], [1.0], [1.0]])).all()

        front_edge = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(1, 2)
        )

        assert (front_edge == np.array([[2.0], [2.0], [2.0]])).all()

        front_edge = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(2, 3)
        )

        assert (front_edge == np.array([[3.0], [3.0], [3.0]])).all()

        extractor = ac.ExtractorSerialFrontEdge(region_list=[(0, 3, 1, 5)])

        front_edge = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(0, 2)
        )

        assert (front_edge == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])).all()

        front_edge = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(1, 4)
        )

        assert (
            front_edge == np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
        ).all()

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edges = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(0, 1)
        )

        assert (front_edges[0] == np.array([[1.0], [1.0], [1.0]])).all()
        assert (front_edges[1] == np.array([[5.0], [5.0], [5.0]])).all()

        front_edges = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(1, 2)
        )

        assert (front_edges[0] == np.array([[2.0], [2.0], [2.0]])).all()
        assert (front_edges[1] == np.array([[6.0], [6.0], [6.0]])).all()

        front_edges = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(2, 3)
        )

        assert (front_edges[0] == np.array([[3.0], [3.0], [3.0]])).all()
        assert (front_edges[1] == np.array([[7.0], [7.0], [7.0]])).all()

        front_edges = extractor.serial_front_edge_arrays_from(
            array=serial_array, columns=(0, 3)
        )

        assert (
            front_edges[0]
            == np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        ).all()

        assert (
            front_edges[1]
            == np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
        ).all()

    def test__array_2d_list_from__include_masking(self, serial_masked_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edges = extractor.serial_front_edge_arrays_from(
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
                [[False, False, False], [False, True, False], [False, False, True]]
            )
        ).all()

    def test__stacked_array_2d_from(self, serial_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        stacked_front_edges = extractor.serial_front_edge_stacked_array_from(
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

    def test__stacked_array_2d_from__include_masking(self, serial_masked_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        stacked_front_edges = extractor.serial_front_edge_stacked_array_from(
            array=serial_masked_array, columns=(0, 3)
        )

        print(stacked_front_edges)

        assert (
            stacked_front_edges
            == np.array([[3.0, 4.0, 5.0], [3.0, 2.0, 5.0], [3.0, 4.0, 3.0]])
        ).all()
        assert (
            stacked_front_edges.mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__binned_over_columns_array_1d_from_columns_from(self, serial_array):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edge_line = extractor.serial_front_edge_line_binned_over_rows_from(
            array=serial_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 5.0])).all()

    def test__binned_over_columns_array_1d_from_columns__include_masking(
        self, serial_masked_array
    ):

        extractor = ac.ExtractorSerialFrontEdge(
            region_list=[(0, 3, 1, 4), (0, 3, 5, 8)]
        )

        front_edge_line = extractor.serial_front_edge_line_binned_over_rows_from(
            array=serial_masked_array, columns=(0, 3)
        )

        assert (front_edge_line == np.array([3.0, 4.0, 13.0 / 3.0])).all()


class TestAbstractLayout2DCI(object):
    def test__total_rows_minimum(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(1, 2, 0, 1)]
        )

        assert layout.total_rows_min == 1

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(1, 3, 0, 1)]
        )

        assert layout.total_rows_min == 2

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(1, 2, 0, 1), (3, 4, 0, 1)]
        )

        assert layout.total_rows_min == 1

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(1, 2, 0, 1), (3, 5, 0, 1)]
        )

        assert layout.total_rows_min == 1

    def test__total_columns_minimum(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 1, 1, 2)]
        )

        assert layout.total_columns_min == 1

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 1, 1, 3)]
        )

        assert layout.total_columns_min == 2

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(0, 1, 1, 2), (0, 1, 3, 4)]
        )

        assert layout.total_columns_min == 1

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 5), normalization=1.0, region_list=[(0, 1, 1, 2), (0, 1, 3, 5)]
        )

        assert layout.total_columns_min == 1

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

    def test__with_extracted_region_list__region_list_are_extracted_correctly(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 2, 0, 2)]
        )

        layout_extracted = layout.with_extracted_region_list(
            extraction_region=ac.Region2D((0, 2, 0, 2))
        )

        assert layout_extracted.region_list == [(0, 2, 0, 2)]

        layout_extracted = layout.with_extracted_region_list(
            extraction_region=ac.Region2D((0, 1, 0, 1))
        )

        assert layout_extracted.region_list == [(0, 1, 0, 1)]

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(10, 5),
            normalization=1.0,
            region_list=[(2, 4, 2, 4), (0, 1, 0, 1)],
        )

        layout_extracted = layout.with_extracted_region_list(
            extraction_region=ac.Region2D((0, 3, 0, 3))
        )

        assert layout_extracted.region_list == [(2, 3, 2, 3), (0, 1, 0, 1)]

        layout_extracted = layout.with_extracted_region_list(
            extraction_region=ac.Region2D((2, 5, 2, 5))
        )

        assert layout_extracted.region_list == [(0, 2, 0, 2)]

        layout_extracted = layout.with_extracted_region_list(
            extraction_region=ac.Region2D((8, 9, 8, 9))
        )

        assert layout_extracted.region_list == None

    def test__frame_with_extracted_region_list_ci_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_region_list_ci_from(frame=frame)

        assert (
            frame_extracted
            == np.array(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 1, 2), (2, 3, 1, 3)],
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_region_list_ci_from(frame=frame)

        assert (
            frame_extracted
            == np.array(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 7.0, 8.0], [0.0, 0.0, 0.0]]
            )
        ).all()

    def test__frame_with_extracted_non_region_list_ci_from(self,):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_non_region_list_ci_from(frame=frame)

        print(frame_extracted)

        assert (
            frame_extracted
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 10.0, 11.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_non_region_list_ci_from(frame=frame)

        assert (
            frame_extracted
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

    def test__frame_with_extracted_parallel_trails_from(self,):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            scans=ac.Scans(serial_prescan=(3, 4, 2, 3), serial_overscan=(3, 4, 0, 1)),
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_parallel_trails_from(frame=frame)

        assert (
            frame_extracted
            == np.array(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=10.0,
            region_list=[(0, 1, 0, 3), (3, 4, 0, 3)],
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
            ],
            scans=ac.Scans(serial_prescan=(1, 2, 0, 3), serial_overscan=(0, 1, 0, 1)),
            pixel_scales=1.0,
        )

        frame_extracted = layout.extract_frame_of_parallel_trails_from(frame=frame)

        assert (
            frame_extracted
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

    def test__parallel_calibration_frame_from(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 3, 0, 3)]
        )

        frame = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
            ],
            pixel_scales=1.0,
        )

        new_array = layout.extract_frame_of_parallel_edges_and_trails_from(
            frame=frame, front_edge_rows=(0, 1)
        )

        assert (
            new_array
            == np.array(
                [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 4, 0, 3)]
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

        new_array = layout.extract_frame_of_parallel_edges_and_trails_from(
            frame=array, trails_rows=(0, 1)
        )

        assert (
            new_array
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

    def test__front_edge_and_trails__2_rows_of_each__new_frame_is_edge_and_trail(self):
        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 4, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ],
            layout=layout,
            pixel_scales=1.0,
        )

        new_array = layout.extract_frame_of_parallel_edges_and_trails_from(
            front_edge_rows=(0, 2), trails_rows=(0, 2)
        )

        assert (
            new_array
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
        assert new_layout.layout.region_list == [(0, 4, 0, 3)]

    def test__front_edge_and_trails__2_region_list__1_row_of_each__new_frame_is_edge_and_trail(
        self,
    ):
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
                [15.0, 16.0, 17.0],
            ],
            layout=layout,
            pixel_scales=1.0,
        )

        new_array = layout.extract_frame_of_parallel_edges_and_trails_from(
            front_edge_rows=(0, 1), trails_rows=(0, 1)
        )

        assert (
            new_array
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
        assert new_layout.layout.region_list == [(0, 1, 0, 3), (3, 4, 0, 3)]


class TestLayout2DCIUniform(object):
    def test__pre_cti_ci_from_shape_native__image_3x3__1_ci_region(self):
        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 2, 0, 2)]
        )

        pre_cti_ci = layout.pre_cti_ci_from(shape_native=(3, 3), pixel_scales=1.0)

        assert (
            pre_cti_ci
            == np.array([[10.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

    def test__pre_cti_ci_from_shape_native__image_3x3__2_region_list_ci(self):
        layout_uni = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=20.0,
            region_list=[(0, 2, 0, 2), (2, 3, 2, 3)],
        )
        image1 = layout_uni.pre_cti_ci_from(shape_native=(3, 3), pixel_scales=1.0)

        assert (
            image1 == np.array([[20.0, 20.0, 0.0], [20.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        ).all()

        layout_uni = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=30.0,
            region_list=[(0, 3, 0, 2), (2, 3, 2, 3)],
        )
        image1 = layout_uni.pre_cti_ci_from(shape_native=(4, 3), pixel_scales=1.0)

        assert (
            image1
            == np.array(
                [
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 0.0],
                    [30.0, 30.0, 30.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__pre_cti_ci_from_shape_native__layout_bigger_than_image_dimensions__raises_error(
        self,
    ):
        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(0, 2, 0, 1)]
        )

        with pytest.raises(exc.Layout2DCIException):
            layout.pre_cti_ci_from(shape_native=(1, 1), pixel_scales=1.0)


class TestLayout2DCINonUniform(object):
    def test__ci_region_from__uniform_column_and_uniform_row__returns_uniform_charge_region(
        self,
    ):
        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=0.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        assert (
            region
            == np.array(
                [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
            )
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            normalization=500.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=0.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        assert (
            region
            == np.array(
                [
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                    [500.0, 500.0, 500.0],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_column_and_uniform_row__returns_region(self):
        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [101.6, 99.4, 99.5], [101.6, 99.4, 99.5]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            normalization=500.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                    [501.6, 499.4, 499.5],
                ]
            )
        ).all()

    def test__ci_region_from__uniform_column_and_non_uniform_row__returns_region(self):
        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[100.0, 100.0, 100.0], [99.3, 99.3, 99.3], [98.9, 98.9, 98.9]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            normalization=500.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [500.0, 500.0, 500.0],
                    [496.5, 496.5, 496.5],
                    [494.5, 494.5, 494.5],
                    [493.1, 493.1, 493.1],
                    [492.0, 492.0, 492.0],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_column_and_non_uniform_row__returns_region(
        self,
    ):
        layout = ac.ci.Layout2DCINonUniform(
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(3, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array([[101.6, 99.4, 99.5], [100.9, 98.7, 98.8], [100.5, 98.3, 98.4]])
        ).all()

        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=500.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(5, 3), ci_seed=1)

        region = np.round(region, 1)

        assert (
            region
            == np.array(
                [
                    [501.6, 499.4, 499.5],
                    [498.2, 495.9, 496.0],
                    [496.1, 493.9, 494.0],
                    [494.7, 492.5, 492.6],
                    [493.6, 491.4, 491.5],
                ]
            )
        ).all()

    def test__ci_region_from__non_uniform_columns_with_large_deviation_value__no_negative_charge_columns_are_generated(
        self,
    ):
        layout = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 1, 0, 1)],
            row_slope=0.0,
            column_sigma=100.0,
        )

        region = layout.ci_region_from_region(region_dimensions=(10, 10), ci_seed=1)

        assert (region > 0).all()

    def test__pre_cti_ci_from__no_non_uniformity__identical_to_uniform_image__one_ci_region(
        self,
    ):
        layout_uni = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=10.0, region_list=[(2, 4, 0, 5)]
        )
        image1 = layout_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            normalization=10.0,
            region_list=[(2, 4, 0, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        image2 = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        assert (image1 == image2).all()

        layout_uni = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=100.0, region_list=[(1, 4, 2, 5)]
        )
        image1 = layout_uni.pre_cti_ci_from(shape_native=(5, 7), pixel_scales=1.0)

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            normalization=100.0,
            region_list=[(1, 4, 2, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        image2 = layout_non_uni.pre_cti_ci_from(shape_native=(5, 7), pixel_scales=1.0)

        assert (image1 == image2).all()

        layout_uni = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 2, 0, 2), (2, 3, 0, 5)],
        )
        image1 = layout_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 2, 0, 2), (2, 3, 0, 5)],
            row_slope=0.0,
            column_sigma=0.0,
        )
        image2 = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        assert (image1 == image2).all()

    def test__pre_cti_ci_from__non_uniformity_in_columns_only__one_ci_region__image_is_correct(
        self,
    ):
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

        image = np.round(image, 1)

        assert (
            image
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
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(1, 4, 1, 4)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
            == np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 101.6, 99.4, 99.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

        layout_non_uni = ac.ci.Layout2DCINonUniform(
            normalization=100.0,
            region_list=[(1, 4, 1, 3), (1, 4, 4, 5)],
            row_slope=0.0,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
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

    def test__pre_cti_ci_from__non_uniformity_in_columns_only__maximum_normalization_input__does_not_simulate_above(
        self,
    ):
        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
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

    def test__pre_cti_ci_from__non_uniformity_in_rows_only__one_ci_region__image_is_correct(
        self,
    ):
        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(0, 3, 0, 3)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (
            image
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
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=0.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (
            image
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

    def test__pre_cti_ci_from__non_uniformity_in_rows_and_columns__two_region_list_ci__image_is_correct(
        self,
    ):
        layout_non_uni = ac.ci.Layout2DCINonUniform(
            shape_2d=(5, 3),
            normalization=100.0,
            region_list=[(1, 5, 1, 4), (0, 5, 4, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(
            shape_native=(5, 5), pixel_scales=1.0, ci_seed=1
        )

        image = np.round(image, 1)

        assert (
            image
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
            normalization=100.0,
            region_list=[(0, 2, 0, 5), (3, 5, 0, 5)],
            row_slope=-0.01,
            column_sigma=1.0,
        )

        image = layout_non_uni.pre_cti_ci_from(shape_native=(5, 5), pixel_scales=1.0)

        image = np.round(image, 1)

        assert (image[0:2, 0:5] == image[3:5, 0:5]).all()


class TestCIRegionFrom:
    def test__region_list_ci_from(self):

        region_list_ci = region_list_ci_from(
            injection_on=10,
            injection_off=10,
            injection_total=1,
            parallel_size=10,
            serial_prescan_size=1,
            serial_size=10,
            serial_overscan_size=1,
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
        )

        assert region_list_ci == [
            (0, 200, 51, 2099),
            (400, 600, 51, 2099),
            (800, 1000, 51, 2099),
            (1200, 1400, 51, 2099),
            (1600, 1800, 51, 2099),
        ]


class TestParallelCalibrationFrame:
    def test__columns_0_to_1__extracts_1_column_left_hand_side_of_array(self):

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
            layout=layout,
            pixel_scales=1.0,
        )

        extracted_array = layout.array_for_parallel_calibration_from(
            array=array, columns=(0, 1)
        )

        assert (extracted_array == np.array([[0.0], [0.0], [0.0], [0.0], [0.0]])).all()
        assert extracted_layout.layout.region_list == [(0, 3, 0, 1)]

    def test__columns_1_to_3__extracts_2_columns_middle_and_right_of_array(self):

        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 5, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            layout=layout,
            pixel_scales=1.0,
        )

        extracted_array = layout.array_for_parallel_calibration_from(columns=(1, 3))

        assert (
            extracted_array
            == np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        ).all()
        assert extracted_layout.layout.region_list == [(0, 5, 0, 2)]

    def test__parallel_extracted_mask(self):
        layout = ac.ci.Layout2DCIUniform(
            shape_2d=(5, 3), normalization=1.0, region_list=[(0, 5, 0, 3)]
        )

        array = ac.Array2D.manual(
            array=[
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            layout=layout,
            pixel_scales=1.0,
        )

        mask = ac.ci.Mask2DCI.unmasked(shape_native=(5, 3), pixel_scales=1.0)

        mask[0, 1] = True

        extracted_mask = layout.mask_for_parallel_calibration_from(mask, columns=(1, 3))

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
