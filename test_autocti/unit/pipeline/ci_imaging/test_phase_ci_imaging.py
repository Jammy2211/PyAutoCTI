from os import path

import numpy as np
import pytest

import autofit as af
import arctic as ac

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
        self, ci_data, clocker
    ):

        phase = PhaseCIImaging(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        results =

        analysis = phase.make_analysis(
            datasets=[ci_data, ci_data, ci_data, ci_data],
            clocker=clocker,
            results=results,
        )

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
