import pytest
from os import path
import os
import shutil

import autofit as af
import autocti as ac
from autofit.non_linear.samples import Sample


def clean(database_file, result_path):
    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


@pytest.fixture(name="model_1d")
def make_model_1d():
    trap_0 = af.Model(ac.TrapInstantCapture)

    trap_list = [trap_0]

    ccd = af.Model(ac.CCDPhase)

    return af.Collection(cti=af.Model(ac.CTI1D, trap_list=trap_list, ccd=ccd))


@pytest.fixture(name="samples_1d")
def make_samples_1d(model_1d):
    trap_0 = ac.TrapInstantCapture(density=0.1, release_timescale=1.0)
    ccd = ac.CCDPhase()

    cti = ac.CTI1D(trap_list=[trap_0], ccd=ccd)

    parameters = [model_1d.prior_count * [1.0], model_1d.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model_1d,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ac.m.MockSamples(
        model=model_1d, sample_list=sample_list, max_log_likelihood_instance=cti
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
    trap_0 = ac.TrapInstantCapture(density=0.1, release_timescale=1.0)
    ccd = ac.CCDPhase()

    cti = ac.CTI2D(parallel_trap_list=[trap_0], parallel_ccd=ccd)

    parameters = [model_2d.prior_count * [1.0], model_2d.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model_2d,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ac.m.MockSamples(
        model=model_2d, sample_list=sample_list, max_log_likelihood_instance=cti
    )
