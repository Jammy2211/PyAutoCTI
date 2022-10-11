import pytest

from autoarray import Array1D
from autocti import Dataset1D


@pytest.fixture(name="flux")
def make_flux():
    return 1234.0


@pytest.fixture(name="pixel_line_dict")
def make_pixel_line_dict(flux):
    return {
        "location": [
            2,
            4,
        ],
        "date": 2453963.778275463,
        "background": 31.30540652532858,
        "flux": flux,
        "data": [
            5.0,
            3.0,
            2.0,
            1.0,
        ],
        "noise": [
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    }


@pytest.fixture(name="size")
def make_size():
    return 10


@pytest.fixture(name="dataset_1d")
def make_dataset_1d(pixel_line_dict, size):
    return Dataset1D.from_pixel_line_dict(
        pixel_line_dict,
        size=size,
    )


def test_parse_data(dataset_1d):
    assert (
        dataset_1d.data
        == Array1D.manual_native(
            [0.0, 0.0, 5.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0], pixel_scales=0.1
        )
    ).all()


def test_parse_noise(dataset_1d):
    assert (
        dataset_1d.noise_map
        == Array1D.manual_native(
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], pixel_scales=0.1
        )
    ).all()


@pytest.fixture(name="pre_cti_data")
def make_pre_cti_data(dataset_1d):
    return dataset_1d.pre_cti_data


def test_pre_cti(pre_cti_data, flux):
    assert (
        pre_cti_data
        == Array1D.manual_native(
            [0.0, 0.0, flux, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], pixel_scales=0.1
        )
    ).all()


@pytest.fixture(name="layout")
def make_layout(dataset_1d):
    return dataset_1d.layout


def test_layout(layout, size):
    assert layout.shape_1d == (size,)


def test_region(layout, pre_cti_data, flux):
    (region,) = layout.region_list
    assert pre_cti_data[region.slice] == flux


def test_float_location():
    pixel_line_dict = {
        "location": [13.0, 0],
        "date": 2453963.758414352,
        "background": 31.30540652532858,
        "flux": 992.075,
        "data": [
            46.933043816970624,
            46.3701649771964,
            48.711080079945226,
            48.11518701894584,
            48.14361054867522,
            48.00110239950205,
            46.16310970820016,
            47.16358109918174,
            46.49024043290878,
            46.863087745614244,
            45.796840479854254,
            45.55346051503446,
            1309.4024294348244,
            48.59285309531201,
            46.63910977027298,
            47.10570908449849,
            48.99711399716423,
            48.20789077340186,
            47.29295786328075,
            48.53875401695496,
            48.920490584648604,
            48.141066131134835,
            47.34273778298907,
            50.227607541836136,
            47.744377164578616,
        ],
        "noise": [
            0.5256967407012534,
            0.4280892624149115,
            0.789855949552933,
            0.6217203894640145,
            0.5786548351791704,
            0.5849704312132465,
            0.4655796354315316,
            0.520450891288638,
            0.46729646303852607,
            0.5451477925803473,
            0.32636264608977256,
            0.2349003033720228,
            3.0453790917701253,
            0.3580557547977746,
            0.28042682627171855,
            0.44804424425163564,
            0.6752575443696331,
            0.4971518347953273,
            0.4332561468089131,
            0.6849637230632679,
            0.5805219047418141,
            0.5610511215490508,
            0.5220946749091355,
            0.7818605598254397,
            0.558755420934289,
        ],
    }
    size = 2068

    Dataset1D.from_pixel_line_dict(
        pixel_line_dict,
        size=size,
    )
