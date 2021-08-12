import autofit as af
import autocti as ac
import pytest
from autocti import exc
from autocti.mock import mock
from autocti.model import result as res


class TestAnalysis:
    def test__parallel_and_serial_checks_raise_exception(self, imaging_ci_7x7):

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI2D,
                parallel_traps=[
                    ac.TrapInstantCapture(density=1.1),
                    ac.TrapInstantCapture(density=1.1),
                ],
                parallel_ccd=ac.CCDPhase(),
            )
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=imaging_ci_7x7,
            clocker=None,
            settings_cti=ac.SettingsCTI2D(parallel_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI2D,
                serial_traps=[
                    ac.TrapInstantCapture(density=1.1),
                    ac.TrapInstantCapture(density=1.1),
                ],
                serial_ccd=ac.CCDPhase(),
            )
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=[imaging_ci_7x7],
            clocker=None,
            settings_cti=ac.SettingsCTI2D(serial_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)
