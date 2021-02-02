from os import path

from autofit.mapper.prior_model import prior_model
import autocti as ac
import numpy as np
import pytest
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from autocti.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__extractions_using_columns_and_rows(self, ci_imaging_7x7):

        ci_imaging_7x7.cosmic_ray_map = None

        phase_ci_imaging_7x7 = PhaseCIImaging(search=mock.MockSearch(name="test_phase"))

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].figures == np.ones(shape=(7, 7))).all()
        assert (
            analysis.masked_ci_datasets[0].figure_noise_map
            == 2.0 * np.ones(shape=(7, 7))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7, 7))
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            settings=ac.SettingsPhaseCIImaging(
                settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                    parallel_columns=(0, 1)
                )
            ),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].figures == np.ones(shape=(7, 1))).all()
        assert (
            analysis.masked_ci_datasets[0].figure_noise_map
            == 2.0 * np.ones(shape=(7, 1))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(7, 1))
        ).all()

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            settings=ac.SettingsPhaseCIImaging(
                settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                    serial_rows=(0, 1)
                )
            ),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        assert (analysis.masked_ci_datasets[0].figures == np.ones(shape=(1, 7))).all()
        assert (
            analysis.masked_ci_datasets[0].figure_noise_map
            == 2.0 * np.ones(shape=(1, 7))
        ).all()
        assert (
            analysis.masked_ci_datasets[0].mask
            == np.full(fill_value=False, shape=(1, 7))
        ).all()

    def test__masks_uses_front_edge_and_trails_parameters(self, ci_imaging_7x7):

        settings_ci_mask = ac.ci.SettingsCIMask(
            parallel_front_edge_rows=(0, 1),
            parallel_trails_rows=(0, 4),
            serial_trails_columns=(1, 2),
            serial_front_edge_columns=(2, 3),
        )

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            settings=ac.SettingsPhaseCIImaging(settings_ci_mask=settings_ci_mask),
        )

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        mask = ac.Mask2D.from_cosmic_ray_map_buffed(
            cosmic_ray_map=ci_imaging_7x7.cosmic_ray_map
        )

        ci_mask = ac.ci.CIMask.masked_front_edges_and_trails_from_ci_frame(
            mask=mask, settings=settings_ci_mask, ci_frame=ci_imaging_7x7.image
        )

        assert (analysis.masked_ci_datasets[0].mask == ci_mask).all()

    def test__noise_scaling_maps_are_setup_correctly(
        self, ci_imaging_7x7, ci_pattern_7x7, parallel_clocker
    ):

        ci_imaging_7x7.cosmic_ray_map = None

        noise_scaling_maps_list_of_ci_regions = [
            ac.ci.CIFrame.ones(
                shape_native=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]
        noise_scaling_maps_list_of_parallel_trails = [
            ac.ci.CIFrame.full(
                fill_value=2.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_trails = [
            ac.ci.CIFrame.full(
                fill_value=3.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_overscan_no_trails = [
            ac.ci.CIFrame.full(
                fill_value=4.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                ci_pattern=ci_pattern_7x7,
            )
        ]

        phase = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            parallel_traps=[prior_model.PriorModel(ac.TrapInstantCapture)],
            parallel_ccd=ac.CCD,
            hyper_noise_scalar_of_ci_regions=ac.ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ac.ci.CIHyperNoiseScalar,
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        assert len(analysis.masked_ci_imagings[0].subplot_noise_scaling_maps) == 2

        assert (
            analysis.masked_ci_imagings[0].subplot_noise_scaling_maps[0]
            == np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].subplot_noise_scaling_maps[1]
            == 2.0 * np.ones((7, 7))
        ).all()

        phase = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            parallel_traps=[prior_model.PriorModel(ac.TrapInstantCapture)],
            parallel_ccd=ac.CCD,
            hyper_noise_scalar_of_parallel_trails=ac.ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_trails=ac.ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_overscan_no_trails=ac.ci.CIHyperNoiseScalar,
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        assert len(analysis.masked_ci_imagings[0].subplot_noise_scaling_maps) == 3

        assert (
            analysis.masked_ci_imagings[0].subplot_noise_scaling_maps[0]
            == 2.0 * np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].subplot_noise_scaling_maps[1]
            == 3.0 * np.ones((7, 7))
        ).all()

        assert (
            analysis.masked_ci_imagings[0].subplot_noise_scaling_maps[2]
            == 4.0 * np.ones((7, 7))
        ).all()
