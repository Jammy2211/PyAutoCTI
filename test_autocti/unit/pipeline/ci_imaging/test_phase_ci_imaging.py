from os import path

import numpy as np
import pytest

import autofit as af
import arctic as ac

import autocti.structures as struct
import autocti.charge_injection as ci
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__extractions_using_columns_and_rows(self, ci_imaging_7x7):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(phase_name="test_phase")

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].image == np.ones(shape=(7, 7))).all()
        assert (
            analysis.masked_ci_datasets[0].noise_map == 2.0 * np.ones(shape=(7, 7))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7, 7))
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(phase_name="test_phase", columns=(0, 1))

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].image == np.ones(shape=(7, 1))).all()
        assert (
            analysis.masked_ci_datasets[0].noise_map == 2.0 * np.ones(shape=(7, 1))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7, 1))
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(phase_name="test_phase", rows=(0, 1))

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].image == np.ones(shape=(1, 7))).all()
        assert (
            analysis.masked_ci_datasets[0].noise_map == 2.0 * np.ones(shape=(1, 7))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(1, 7))
        ).all()

    def test__masks_uses_front_edge_and_trails_parameters(self, ci_imaging_7x7):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase", parallel_front_edge_mask_rows=(0, 1)
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, True, True, True, True, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase", parallel_trails_mask_rows=(0, 1)
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, True, True, True, True, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase", serial_front_edge_mask_columns=(0, 1)
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
                    [False, True, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, True, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            phase_name="test_phase", serial_trails_mask_columns=(0, 1)
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (
            analysis.masked_ci_datasets[0].mask
            == np.array(
                [
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, True, False],
                    [False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False],
                ]
            )
        ).all()

    def test__noise_scaling_maps_are_setup_correctly__ci_regions_and_parallel_trail_scalars(
        self, ci_imaging_7x7, ci_pattern_7x7, parallel_clocker
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        noise_scaling_maps_list_of_ci_regions = [
            ci.CIFrame.ones(
                shape_2d=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]
        noise_scaling_maps_list_of_parallel_trails = [
            ci.CIFrame.full(
                fill_value=2.0,
                shape_2d=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_trails = [
            ci.CIFrame.full(
                fill_value=3.0,
                shape_2d=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_overscan_no_trails = [
            ci.CIFrame.full(
                fill_value=4.0,
                shape_2d=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]

        phase = PhaseCIImaging(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            hyper_noise_scalar_of_ci_regions=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ci.CIHyperNoiseScalar,
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        assert len(analysis.masked_ci_imagings[0].noise_scaling_maps) == 2

        assert (
            analysis.masked_ci_imagings[0].noise_scaling_maps[0] == np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].noise_scaling_maps[1]
            == 2.0 * np.ones((7, 7))
        ).all()

        phase = PhaseCIImaging(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            hyper_noise_scalar_of_parallel_trails=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_trails=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_overscan_no_trails=ci.CIHyperNoiseScalar,
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        assert len(analysis.masked_ci_imagings[0].noise_scaling_maps) == 3

        assert (
            analysis.masked_ci_imagings[0].noise_scaling_maps[0]
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].noise_scaling_maps[1]
            == 3.0 * np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].noise_scaling_maps[2]
            == 4.0 * np.ones((7, 7))
        ).all()
