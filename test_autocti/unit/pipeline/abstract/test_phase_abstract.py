from os import path

import pytest

import autofit as af
import arctic as ac
from autocti.pipeline.phase.dataset import PhaseDataset
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

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
        trap_model = af.PriorModel(ac.Trap)
        phase_dataset_7x7.parallel_traps = [trap_model]
        assert phase_dataset_7x7.parallel_traps == [trap_model]

        ccd_volume_model = af.PriorModel(ac.CCDVolume)
        phase_dataset_7x7.parallel_ccd_volume = ccd_volume_model
        assert phase_dataset_7x7.parallel_ccd_volume == ccd_volume_model

    def test__phase_can_receive_list_of_galaxy_models(self):

        phase_dataset_7x7 = PhaseDataset(
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            non_linear_class=af.MultiNest,
            phase_name="test_phase",
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
    def test_assertion_failure(self, ci_imaging_7x7, mask_7x7):

        phase_dataset_7x7 = PhaseCIImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
        )

        result = phase_dataset_7x7.run(
            dataset=[ci_imaging_7x7], mask=None, results=None
        )
        assert result is not None

        phase_dataset_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_dataset_7x7.make_analysis = make_analysis
        result = phase_dataset_7x7.run(
            dataset=imaging_7x7, results=None, mask=None, positions=None
        )
        assert result is not None
