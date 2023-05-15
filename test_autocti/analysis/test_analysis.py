import pytest

import autofit as af
import autocti as ac
from autocti import exc


def test__parallel_and_serial_checks_raise_exception(imaging_ci_7x7):
    model = af.Collection(
        cti=af.Model(
            ac.CTI2D,
            parallel_trap_list=[
                ac.TrapInstantCapture(density=1.1),
                ac.TrapInstantCapture(density=1.1),
            ],
            parallel_ccd=ac.CCDPhase(),
        )
    )

    analysis = ac.AnalysisImagingCI(
        dataset=imaging_ci_7x7,
        clocker=ac.Clocker2D(),
        settings_cti=ac.SettingsCTI2D(parallel_total_density_range=(1.0, 2.0)),
    )

    instance = model.instance_from_prior_medians()

    with pytest.raises(exc.PriorException):
        analysis.log_likelihood_function(instance=instance)

    model = af.Collection(
        cti=af.Model(
            ac.CTI2D,
            serial_trap_list=[
                ac.TrapInstantCapture(density=1.1),
                ac.TrapInstantCapture(density=1.1),
            ],
            serial_ccd=ac.CCDPhase(),
        )
    )

    analysis = ac.AnalysisImagingCI(
        dataset=[imaging_ci_7x7],
        clocker=ac.Clocker2D(),
        settings_cti=ac.SettingsCTI2D(serial_total_density_range=(1.0, 2.0)),
    )

    instance = model.instance_from_prior_medians()

    with pytest.raises(exc.PriorException):
        analysis.log_likelihood_function(instance=instance)
