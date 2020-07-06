import arctic as ac
from autofit.mapper.prior_model import prior_model
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from autocti.pipeline.phase.dataset import PhaseDataset
from test_autocti import mock

import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestModel:
    def test__set_instances(self, phase_dataset_7x7):
        trap = ac.Trap()
        phase_dataset_7x7.parallel_traps = [trap]
        assert phase_dataset_7x7.model.parallel_traps == [trap]

    def test__set_models(self, phase_dataset_7x7):
        trap_model = prior_model.PriorModel(ac.Trap)
        phase_dataset_7x7.parallel_traps = [trap_model]
        assert phase_dataset_7x7.parallel_traps == [trap_model]

        ccd_volume_model = prior_model.PriorModel(ac.CCDVolume)
        phase_dataset_7x7.parallel_ccd_volume = ccd_volume_model
        assert phase_dataset_7x7.parallel_ccd_volume == ccd_volume_model

    def test__phase_can_receive_model_objects(self):

        phase_dataset_7x7 = PhaseDataset(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            search=mock.MockSearch(),
        )

        parallel_trap = phase_dataset_7x7.model.parallel_traps[0]
        parallel_ccd_volume = phase_dataset_7x7.model.parallel_ccd_volume
        serial_trap = phase_dataset_7x7.model.serial_traps[0]
        serial_ccd_volume = phase_dataset_7x7.model.serial_ccd_volume

        arguments = {
            parallel_trap.density: 0.1,
            parallel_trap.lifetime: 0.2,
            parallel_ccd_volume.well_max_height: 0.3,
            parallel_ccd_volume.well_notch_depth: 0.4,
            parallel_ccd_volume.well_fill_beta: 0.5,
            serial_trap.density: 0.6,
            serial_trap.lifetime: 0.7,
            serial_ccd_volume.well_max_height: 0.8,
            serial_ccd_volume.well_notch_depth: 0.9,
            serial_ccd_volume.well_fill_beta: 1.0,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments=arguments)

        assert instance.parallel_traps[0].density == 0.1
        assert instance.parallel_traps[0].lifetime == 0.2
        assert instance.parallel_ccd_volume.well_max_height == 0.3
        assert instance.parallel_ccd_volume.well_notch_depth == 0.4
        assert instance.parallel_ccd_volume.well_fill_beta == 0.5
        assert instance.serial_traps[0].density == 0.6
        assert instance.serial_traps[0].lifetime == 0.7
        assert instance.serial_ccd_volume.well_max_height == 0.8
        assert instance.serial_ccd_volume.well_notch_depth == 0.9
        assert instance.serial_ccd_volume.well_fill_beta == 1.0


class TestSetup:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, ci_imaging_7x7, mask_7x7, parallel_clocker):

        phase_dataset_7x7 = PhaseCIImaging(
            phase_name="phase_name",
            parallel_traps=[ac.Trap()],
            parallel_ccd_volume=ac.CCDVolume(),
            search=mock.MockSearch(),
        )

        result = phase_dataset_7x7.run(
            datasets=[ci_imaging_7x7], clocker=parallel_clocker, results=None
        )
        assert result is not None
