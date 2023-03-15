import autocti as ac


def test__median_list_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 0, 5)])

    # Extracts [0.0, 1.0, 2.0] of every injection in `serial_array` and takes median over these values.

    median_list = extract.median_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )

    assert median_list == [1.0, 1.0, 1.0]

    # Extend pixels to extract [0.0, 1.0, 2.0, 3.0, 4.0]

    median_list = extract.median_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 5))
    )
    assert median_list == [2.0, 2.0, 2.0]

    extract = ac.Extract2DSerialFPR(region_list=[(0, 2, 0, 4), (0, 2, 6, 10)])

    median_list = extract.median_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 4))
    )
    assert median_list == [4.5, 4.5]

    median_list = extract.median_list_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels=(0, 4))
    )

    assert median_list == [4.5, 5.0]


def test__median_list_from__pixels_from_end(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialFPR(
        shape_2d=serial_array.shape_native, region_list=[(0, 3, 0, 5)]
    )

    # Extend pixels to extract [0.0, 1.0, 2.0, 3.0, 4.0]

    median_list = extract.median_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=5)
    )
    assert median_list == [2.0, 2.0, 2.0]

    extract = ac.Extract2DSerialFPR(
        shape_2d=serial_array.shape_native, region_list=[(0, 2, 0, 4), (0, 2, 6, 10)]
    )

    median_list = extract.median_list_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=4)
    )
    assert median_list == [4.5, 4.5]

    median_list = extract.median_list_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels_from_end=4)
    )

    assert median_list == [4.5, 5.0]


def test__median_list_of_lists_from(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 0, 5)])

    # Extracts [0.0, 1.0, 2.0] of every injection in `serial_array` and takes median over these values.

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 3))
    )
    assert median_list_of_lists[0] == [1.0, 1.0, 1.0]

    # Extend pixels to extract [0.0, 1.0, 2.0, 3.0, 4.0]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 5))
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]

    extract = ac.Extract2DSerialFPR(region_list=[(0, 2, 0, 4), (0, 2, 6, 10)])

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_array, settings=ac.SettingsExtract(pixels=(0, 4))
    )

    assert median_list_of_lists[0] == [1.5, 1.5]
    assert median_list_of_lists[1] == [7.5, 7.5]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels=(0, 4))
    )
    assert median_list_of_lists[0] == [1.5, 1.0]
    assert median_list_of_lists[1] == [7.5, 8.0]


def test__median_list_of_lists_from__pixels_from_end(serial_array, serial_masked_array):

    extract = ac.Extract2DSerialFPR(region_list=[(0, 3, 0, 5)])

    # Extend pixels to extract [0.0, 1.0, 2.0, 3.0, 4.0]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=5)
    )
    assert median_list_of_lists[0] == [2.0, 2.0, 2.0]

    extract = ac.Extract2DSerialFPR(region_list=[(0, 2, 0, 4), (0, 2, 6, 10)])

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_array, settings=ac.SettingsExtract(pixels_from_end=4)
    )

    assert median_list_of_lists[0] == [1.5, 1.5]
    assert median_list_of_lists[1] == [7.5, 7.5]

    median_list_of_lists = extract.median_list_of_lists_from(
        array=serial_masked_array, settings=ac.SettingsExtract(pixels_from_end=4)
    )
    assert median_list_of_lists[0] == [1.5, 1.0]
    assert median_list_of_lists[1] == [7.5, 8.0]
