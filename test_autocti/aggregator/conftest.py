import pytest
from os import path
import os
import shutil

from autoconf import conf
import autofit as af
import autocti as ac
from autofit.non_linear.samples import Sample


@pytest.fixture(autouse=True)
def set_test_mode():
    os.environ["PYAUTOFIT_TEST_MODE"] = "1"
    yield
    del os.environ["PYAUTOFIT_TEST_MODE"]


def clean(database_file):
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    if path.exists(database_sqlite):
        os.remove(database_sqlite)

    result_path = path.join(conf.instance.output_path, database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


def aggregator_from(database_file, analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)

    clean(database_file=database_file)

    search = ac.m.MockSearch(
        samples=samples, result=ac.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=database_file)
    search.fit(model=model, analysis=analysis)

    database_file = path.join(conf.instance.output_path, f"{database_file}.sqlite")

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    return agg


@pytest.fixture(name="model_1d")
def make_model_1d():
    trap_0 = af.Model(ac.TrapInstantCapture)

    trap_list = [trap_0]

    ccd = af.Model(ac.CCDPhase)

    return af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))


@pytest.fixture(name="samples_1d")
def make_samples_1d(model_1d):
    parameters = [model_1d.prior_count * [1.0], model_1d.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model_1d,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ac.m.MockSamples(
        model=model_1d,
        sample_list=sample_list,
        prior_means=[1.0] * model_1d.prior_count,
    )


@pytest.fixture(name="model_2d")
def make_model_2d():
    trap_0 = af.Model(ac.TrapInstantCapture)

    trap_list = [trap_0]

    ccd = af.Model(ac.CCDPhase)

    return af.Collection(
        cti=af.Model(ac.CTI2D, parallel_trap_list=trap_list, parallel_ccd=ccd)
    )


@pytest.fixture(name="samples_2d")
def make_samples_2d(model_2d):
    parameters = [model_2d.prior_count * [1.0], model_2d.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model_2d,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ac.m.MockSamples(
        model=model_2d,
        sample_list=sample_list,
        prior_means=[1.0] * model_2d.prior_count,
    )
