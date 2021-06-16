import numpy as np
import pytest
import autocti as ac
from autocti import exc


@pytest.fixture(name="array")
def make_array():
    return ac.Array1D.manual_native(
        array=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], pixel_scales=1.0
    )


@pytest.fixture(name="masked_array")
def make_masked_array(array):

    mask = ac.Mask1D.manual(
        mask=[False, False, True, False, False, True, False, False, True],
        pixel_scales=1.0,
    )

    return ac.Array1D.manual_mask(array=array.native, mask=mask)


class TestAbstractExtractor:
    def test__total_pixels_minimum(self):

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 2)])

        assert layout.total_pixels_min == 1

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3)])

        assert layout.total_pixels_min == 2

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3), (0, 5)])

        assert layout.total_pixels_min == 2

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 3), (4, 5)])

        assert layout.total_pixels_min == 1

    def test__total_pixel_spacing_min(self):

        layout = ac.Extractor1DFrontEdge(region_list=[(1, 2)])

        assert layout.total_pixels_min == 1


class TestExtractorFrontEdge:
    def test__array_1d_list_from(self, array, masked_array):

        extractor = ac.Extractor1DFrontEdge(region_list=[(1, 4)])

        front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
        assert (front_edge_list[0] == np.array([1.0])).all()

        front_edge = extractor.array_1d_list_from(array=array, pixels=(2, 3))
        assert (front_edge[0] == np.array([3.0])).all()

        extractor = ac.Extractor1DFrontEdge(region_list=[(1, 4), (5, 8)])

        front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
        assert (front_edge_list[0] == np.array([[1.0]])).all()
        assert (front_edge_list[1] == np.array([[5.0]])).all()

        front_edge_list = extractor.array_1d_list_from(array=array, pixels=(2, 3))
        assert (front_edge_list[0] == np.array([[3.0]])).all()
        assert (front_edge_list[1] == np.array([[7.0]])).all()

        front_edge_list = extractor.array_1d_list_from(array=array, pixels=(0, 3))
        assert (front_edge_list[0] == np.array([1.0, 2.0, 3.0])).all()
        assert (front_edge_list[1] == np.array([5.0, 6.0, 7.0])).all()

        front_edge_list = extractor.array_1d_list_from(
            array=masked_array, pixels=(0, 3)
        )

        assert (front_edge_list[0].mask == np.array([False, True, False])).all()

    def test__stacked_array_1d_from(self, array, masked_array):

        extractor = ac.Extractor1DFrontEdge(region_list=[(1, 4), (5, 8)])

        stacked_front_edges = extractor.stacked_array_1d_from(
            array=array, pixels=(0, 3)
        )

        assert (stacked_front_edges == np.array([3.0, 4.0, 5.0])).all()

        extractor = ac.Extractor1DFrontEdge(region_list=[(1, 3), (5, 8)])

        stacked_front_edges = extractor.stacked_array_1d_from(
            array=array, pixels=(0, 2)
        )

        assert (stacked_front_edges == np.array([3.0, 4.0])).all()

        stacked_front_edges = extractor.stacked_array_1d_from(
            array=masked_array, pixels=(0, 2)
        )

        assert (stacked_front_edges == np.ma.array([1.0, 6.0])).all()
        assert (stacked_front_edges.mask == np.ma.array([False, False])).all()


class TestExtractorTrails:
    def test__array_1d_list_from(self, array, masked_array):

        extractor = ac.Extractor1DTrails(region_list=[(1, 3)])

        trails_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
        assert (trails_list == np.array([3.0])).all()

        trails_list = extractor.array_1d_list_from(array=array, pixels=(2, 3))
        assert (trails_list == np.array([5.0])).all()

        trails_list = extractor.array_1d_list_from(array=array, pixels=(1, 4))
        assert (trails_list == np.array([4.0, 5.0, 6.0])).all()

        extractor = ac.Extractor1DTrails(region_list=[(1, 3), (4, 6)])

        trails_list = extractor.array_1d_list_from(array=array, pixels=(0, 1))
        assert (trails_list[0] == np.array([3.0])).all()
        assert (trails_list[1] == np.array([6.0])).all()

        trails_list = extractor.array_1d_list_from(array=array, pixels=(0, 2))
        assert (trails_list[0] == np.array([3.0, 4.0])).all()
        assert (trails_list[1] == np.array([6.0, 7.0])).all()

        trails_list = extractor.array_1d_list_from(array=masked_array, pixels=(0, 2))

        assert (trails_list[0].mask == np.array([False, False])).all()

        assert (trails_list[1].mask == np.array([False, False])).all()

    def test__stacked_array_1d_from(self, array, masked_array):

        extractor = ac.Extractor1DTrails(region_list=[(1, 3), (5, 7)])

        stacked_trails = extractor.stacked_array_1d_from(array=array, pixels=(0, 2))

        assert (stacked_trails == np.array([5.0, 6.0])).all()

        stacked_trails = extractor.stacked_array_1d_from(
            array=masked_array, pixels=(0, 2)
        )

        assert (stacked_trails == np.array([5.0, 4.0])).all()
        assert (stacked_trails.mask == np.array([False, False])).all()


class TestAbstractLayout1DLine:
    def test__trail_size_to_array_edge(self):

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=1.0, region_list=[ac.Region1D(region=(0, 3))]
        )

        assert layout.trail_size_to_array_edge == 2

        layout = ac.Layout1DLineUniform(
            shape_1d=(7,), normalization=1.0, region_list=[ac.Region1D(region=(0, 3))]
        )

        assert layout.trail_size_to_array_edge == 4

        layout = ac.Layout1DLineUniform(
            shape_1d=(15,),
            normalization=1.0,
            region_list=[
                ac.Region1D(region=(0, 2)),
                ac.Region1D(region=(5, 8)),
                ac.Region1D(region=(11, 14)),
            ],
        )

        assert layout.trail_size_to_array_edge == 1

        layout = ac.Layout1DLineUniform(
            shape_1d=(20,),
            normalization=1.0,
            region_list=[
                ac.Region1D(region=(0, 2)),
                ac.Region1D(region=(5, 8)),
                ac.Region1D(region=(11, 14)),
            ],
        )

        assert layout.trail_size_to_array_edge == 6

    def test__array_1d_of_regions_from(self):

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=10.0, region_list=[(0, 3)]
        )

        array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

        array_extracted = layout.array_1d_of_regions_from(array=array)

        assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=10.0, region_list=[(0, 1), (2, 3)]
        )

        array_extracted = layout.array_1d_of_regions_from(array=array)

        assert (array_extracted == np.array([0.0, 0.0, 2.0, 0.0])).all()

    def test__array_1d_of_non_regions_from(self):

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=10.0, region_list=[(0, 3)]
        )

        array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

        array_extracted = layout.array_1d_of_non_regions_from(array=array)

        assert (array_extracted == np.array([0.0, 0.0, 0.0, 3.0])).all()

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=10.0, region_list=[(0, 1), (3, 4)]
        )

        array_extracted = layout.array_1d_of_non_regions_from(array=array)

        assert (array_extracted == np.array([0.0, 1.0, 2.0, 0.0])).all()

    def test__array_1d_of_trails_from(self):

        layout = ac.Layout1DLineUniform(
            shape_1d=(4,),
            normalization=10.0,
            region_list=[(0, 2)],
            prescan=(0, 1),
            overscan=(0, 1),
        )

        array = ac.Array1D.manual_native(array=[0.0, 1.0, 2.0, 3.0], pixel_scales=1.0)

        array_extracted = layout.array_1d_of_trails_from(array=array)

        assert (array_extracted == np.array([0.0, 0.0, 2.0, 3.0])).all()

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,),
            normalization=10.0,
            region_list=[(0, 2)],
            prescan=(2, 3),
            overscan=(3, 4),
        )

        array_extracted = layout.array_1d_of_trails_from(array=array)

        assert (array_extracted == np.array([0.0, 0.0, 0.0, 0.0])).all()

        layout = ac.Layout1DLineUniform(
            shape_1d=(4,),
            normalization=10.0,
            region_list=[(0, 1), (2, 3)],
            prescan=(2, 3),
            overscan=(2, 3),
        )

        array_extracted = layout.array_1d_of_trails_from(array=array)

        assert (array_extracted.native == np.array([0.0, 1.0, 0.0, 3.0])).all()

    def test__array_1d_of_edges_and_trails_from(self, array):

        layout = ac.Layout1DLineUniform(
            shape_1d=(10,), normalization=10.0, region_list=[(0, 4)]
        )

        extracted_array = layout.array_1d_of_edges_and_trails_from(
            array=array, front_edge_rows=(0, 2), trails_rows=(0, 2)
        )

        assert (
            extracted_array
            == np.array([0.0, 1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        ).all()

        layout = ac.Layout1DLineUniform(
            shape_1d=(10,), normalization=10.0, region_list=[(0, 1), (3, 4)]
        )

        extracted_array = layout.array_1d_of_edges_and_trails_from(
            array=array, front_edge_rows=(0, 1), trails_rows=(0, 1)
        )

        assert (
            extracted_array
            == np.array([0.0, 1.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ).all()


class TestLayout2DCIUniform(object):
    def test__pre_cti_image_from(self):

        layout = ac.Layout1DLineUniform(
            shape_1d=(5,), normalization=10.0, region_list=[(0, 2)]
        )

        pre_cti_image = layout.pre_cti_image_from(shape_native=(3,), pixel_scales=1.0)

        assert (pre_cti_image.native == np.array([10.0, 10.0, 0.0])).all()
