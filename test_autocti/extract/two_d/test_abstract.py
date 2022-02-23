import autocti as ac


def test__total_rows_minimum():

    layout = ac.Extract2DParallelFPR(region_list=[(1, 2, 0, 1)])

    assert layout.total_rows_min == 1

    layout = ac.Extract2DParallelFPR(region_list=[(1, 3, 0, 1)])

    assert layout.total_rows_min == 2

    layout = ac.Extract2DParallelFPR(region_list=[(1, 2, 0, 1), (3, 4, 0, 1)])

    assert layout.total_rows_min == 1

    layout = ac.Extract2DParallelFPR(region_list=[(1, 2, 0, 1), (3, 5, 0, 1)])

    assert layout.total_rows_min == 1


def test__total_columns_minimum():

    layout = ac.Extract2DParallelFPR(region_list=[(0, 1, 1, 2)])

    assert layout.total_columns_min == 1

    layout = ac.Extract2DParallelFPR(region_list=[(0, 1, 1, 3)])

    assert layout.total_columns_min == 2

    layout = ac.Extract2DParallelFPR(region_list=[(0, 1, 1, 2), (0, 1, 3, 4)])

    assert layout.total_columns_min == 1

    layout = ac.Extract2DParallelFPR(region_list=[(0, 1, 1, 2), (0, 1, 3, 5)])

    assert layout.total_columns_min == 1
