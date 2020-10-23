from os import path

import numpy as np
import pytest
import autocti as ac
from autocti.pipeline.phase.extensions import HyperNoisePhase
from autocti.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_noise_phase(self):

        phase = ac.PhaseCIImaging(search=mock.MockSearch(name="test_phase"))

        phase_extended = phase.extend_with_hyper_noise_phases()
        assert type(phase_extended.hyper_phases[0]) == HyperNoisePhase


class TestMakeAnalysis:
    def test__masks_with_no_cosmic_rays_or_other_inputs__is_all_false(
        self, ci_imaging_7x7
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = ac.PhaseCIImaging(
            search=mock.MockSearch(name="test_phase")
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7, 7))
        ).all()

    def test__ci_imaging_with_cosmic_ray_map_uses_it_to_make_mask_with_trails(
        self, ci_imaging_7x7, ci_pattern_7x7
    ):

        cosmic_ray_map = ac.ci.CIFrame.full(
            fill_value=False,
            shape_2d=(7, 7),
            ci_pattern=ci_pattern_7x7,
            pixel_scales=1.0,
        )
        cosmic_ray_map[1, 1] = True

        ci_imaging_7x7.cosmic_ray_map = cosmic_ray_map

        phase_ci_imaging_7x7 = ac.PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            settings=ac.SettingsPhaseCIImaging(
                settings_mask=ac.SettingsMask(
                    cosmic_ray_serial_buffer=0,
                    cosmic_ray_parallel_buffer=0,
                    cosmic_ray_diagonal_buffer=0,
                )
            ),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        cosmic_ray_map = ac.ci.CIFrame.full(
            fill_value=False,
            shape_2d=(7, 7),
            ci_pattern=ci_pattern_7x7,
            pixel_scales=1.0,
        )
        cosmic_ray_map[1, 1] = True

        ci_imaging_7x7.cosmic_ray_map = cosmic_ray_map

        phase_ci_imaging_7x7 = ac.PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            settings=ac.SettingsPhaseCIImaging(
                settings_mask=ac.SettingsMask(
                    cosmic_ray_serial_buffer=2,
                    cosmic_ray_parallel_buffer=1,
                    cosmic_ray_diagonal_buffer=1,
                )
            ),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array(
                [
                    [False, True, True, False, False, False, False],
                    [False, True, True, True, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()
