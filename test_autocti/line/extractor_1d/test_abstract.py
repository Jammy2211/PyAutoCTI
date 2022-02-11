import autocti as ac


def test__total_pixels_minimum():
    layout = ac.Extractor1DFPR(region_list=[(1, 2)])

    assert layout.total_pixels_min == 1

    layout = ac.Extractor1DFPR(region_list=[(1, 3)])

    assert layout.total_pixels_min == 2

    layout = ac.Extractor1DFPR(region_list=[(1, 3), (0, 5)])

    assert layout.total_pixels_min == 2

    layout = ac.Extractor1DFPR(region_list=[(1, 3), (4, 5)])

    assert layout.total_pixels_min == 1


def test__total_pixel_spacing_min():
    layout = ac.Extractor1DFPR(region_list=[(1, 2)])

    assert layout.total_pixels_min == 1
