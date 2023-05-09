import copy
from os import path
import pytest
from autocti import plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_fit_ci_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "fit"
    )


def test__text_manual_dict(fit_ci_7x7):
    fit_ci_7x7 = copy.deepcopy(fit_ci_7x7)
    fit_ci_7x7.dataset.settings_dict = {"hello": 2.0, "hi": 3.0}

    fit_ci_plotter = aplt.FitImagingCIPlotter(fit=fit_ci_7x7)

    assert fit_ci_plotter.text_manual_dict_from(region="eper") == {
        "FPR (e-)": 1.0,
        "hello": 2.0,
        "hi": 3.0,
    }
    assert fit_ci_plotter.text_manual_dict_from(region="fpr") == {
        "hello": 2.0,
        "hi": 3.0,
    }
