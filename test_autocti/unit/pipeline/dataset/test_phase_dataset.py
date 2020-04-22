from os import path

import numpy as np
import pytest

from autocti import charge_injection as ci
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_and_pixelizations(self):

        phase_no_pixelization = al.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO, phase_name="test_phase"
        )

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_no_pixelization

        # This phase does not have a pixelization, so even though inversion=True it will not be extended

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert phase_extended == phase_no_pixelization

        phase_with_pixelization = al.PhaseImaging(
            galaxies=dict(
                source=al.GalaxyModel(
                    redshift=0.5,
                    pixelization=al.pix.Rectangular,
                    regularization=al.reg.Constant,
                )
            ),
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.InversionPhase
        assert type(phase_extended.hyper_phases[1]) == al.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True, hyper_galaxy_phase_first=True
        )
        assert type(phase_extended.hyper_phases[0]) == al.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == al.InversionPhase


class TestMakeAnalysis:

    def test__masks_with_no_cosmic_rays_or_other_inputs__is_all_false(
            self, ci_imaging_7x7
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7,7))
        ).all()

    def test__ci_imaging_with_cosmic_ray_map_uses_it_to_make_mask_with_trails(
        self, ci_imaging_7x7, ci_pattern_7x7
    ):

        cosmic_ray_map = ci.CIFrame.full(fill_value=False, shape_2d=(7,7), ci_pattern=ci_pattern_7x7)
        cosmic_ray_map[1,1] = True

        ci_imaging_7x7.cosmic_ray_map = cosmic_ray_map

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            cosmic_ray_serial_buffer=0,
            cosmic_ray_parallel_buffer=0,
            cosmic_ray_diagonal_buffer=0,
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, False, False, False, False, False, False],
                         [False, True, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

        cosmic_ray_map = ci.CIFrame.full(fill_value=False, shape_2d=(7,7), ci_pattern=ci_pattern_7x7)
        cosmic_ray_map[1,1] = True

        ci_imaging_7x7.cosmic_ray_map = cosmic_ray_map

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            cosmic_ray_serial_buffer=2,
            cosmic_ray_parallel_buffer=1,
            cosmic_ray_diagonal_buffer=1,
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, True, True, False, False, False, False],
                         [False, True, True, True, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

    def test__masks_uses_front_edge_and_trails_parameters(
            self, ci_imaging_7x7
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            parallel_front_edge_mask_rows=(0, 1),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, False, False, False, False, False, False],
                         [False, True, True, True, True, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            parallel_trails_mask_rows=(0, 1),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, True, True, True, True, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            serial_front_edge_mask_columns=(0, 1),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, False, False, False, False, False, False],
                         [False, True, False, False, False, False, False],
                         [False, True, False, False, False, False, False],
                         [False, True, False, False, False, False, False],
                         [False, True, False, False, False, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase",
            serial_trails_mask_columns=(0, 1),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(datasets=[ci_imaging_7x7], cti_settings=None)

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array([[False, False, False, False, False, False, False],
                         [False, False, False, False, False, True, False],
                         [False, False, False, False, False, True, False],
                         [False, False, False, False, False, True, False],
                         [False, False, False, False, False, True, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]])
        ).all()

class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.GalaxiesMockAnalysis(1, 1)

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None, positions=None
        )
        assert result is not None

        phase_imaging_7x7 = al.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None, positions=None
        )
        assert result is not None

        class CustomPhase(al.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.lens.light = al.lp.EllipticalLightProfile()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=al.Galaxy(light=al.lp.EllipticalLightProfile, redshift=1)
            ),
        )
        phase_imaging_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_imaging_7x7.run(data_type=imaging_7x7, results=None, mask=None, positions=None)
