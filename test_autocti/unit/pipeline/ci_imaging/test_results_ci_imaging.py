from os import path

import numpy as np
import pytest
import autofit as af

import arctic as ac
import autocti.charge_injection as ci
from autocti.pipeline.phase.dataset.result import Result
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__fits_to_extracted_and_full_datasets_available(
        self, ci_imaging_7x7, parallel_clocker
    ):

        phase_ci_imaging_7x7 = PhaseCIImaging(
            non_linear_class=mock_pipeline.MockNLO,
            columns=(0, 1),
            phase_name="test_phase_2",
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(),
        )

        assert isinstance(result, Result)
        assert (
            result.max_log_likelihood_fits[0].mask
            == np.full(fill_value=False, shape=(7, 1))
        ).all()

        assert isinstance(result, Result)
        assert (
            result.max_log_likelihood_full_fits[0].mask
            == np.full(fill_value=False, shape=(7, 7))
        ).all()

    def test__noise_scaling_maps_list_of_result__are_correct(
        self, ci_imaging_7x7, parallel_clocker, ci_pattern_7x7
    ):

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

        phase_ci_imaging_7x7 = PhaseCIImaging(
            non_linear_class=mock_pipeline.MockNLO,
            hyper_noise_scalar_of_ci_regions=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_trails=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_overscan_no_trails=ci.CIHyperNoiseScalar,
            phase_name="test_phase_2",
        )

        result = phase_ci_imaging_7x7.run(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        ci_post_cti = parallel_clocker.add_cti(image=ci_imaging_7x7.ci_pre_cti)

        mask = ci.CIMask.unmasked(
            shape_2d=ci_imaging_7x7.shape_2d, pixel_scales=ci_imaging_7x7.pixel_scales
        )
        masked_ci_imaging_7x7 = ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[
                noise_scaling_maps_list_of_ci_regions[0],
                noise_scaling_maps_list_of_parallel_trails[0],
                noise_scaling_maps_list_of_serial_trails[0],
                noise_scaling_maps_list_of_serial_overscan_no_trails[0],
            ],
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[
                ci.CIHyperNoiseScalar(scale_factor=0.0),
                ci.CIHyperNoiseScalar(scale_factor=0.0),
                ci.CIHyperNoiseScalar(scale_factor=0.0),
                ci.CIHyperNoiseScalar(scale_factor=0.0),
            ],
        )

        assert result.noise_scaling_maps_list_of_ci_regions[0] == pytest.approx(
            fit.chi_squared_map.ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_parallel_trails[0] == pytest.approx(
            fit.chi_squared_map.parallel_non_ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_trails[0] == pytest.approx(
            fit.chi_squared_map.serial_trails_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_overscan_no_trails[
            0
        ] == pytest.approx(fit.chi_squared_map.serial_overscan_no_trails_frame, 1.0e-2)

        assert result.noise_scaling_maps_list_of_ci_regions[0][1, 1] == pytest.approx(
            20.25, 1.0e-4
        )
        assert result.noise_scaling_maps_list_of_parallel_trails[0][
            1, 1
        ] == pytest.approx(0.0, 1.0e-4)
        assert result.noise_scaling_maps_list_of_serial_trails[0][
            1, 1
        ] == pytest.approx(0.0, 1.0e-4)
        assert result.noise_scaling_maps_list_of_serial_overscan_no_trails[0][
            1, 1
        ] == pytest.approx(0.0, 1.0e-4)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[
                ci.CIHyperNoiseScalar(scale_factor=1.0),
                ci.CIHyperNoiseScalar(scale_factor=1.0),
                ci.CIHyperNoiseScalar(scale_factor=1.0),
                ci.CIHyperNoiseScalar(scale_factor=1.0),
            ],
        )

        assert result.noise_scaling_maps_list_of_ci_regions[0] != pytest.approx(
            fit.chi_squared_map.ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_parallel_trails[0] != pytest.approx(
            fit.chi_squared_map.parallel_non_ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_trails[0] != pytest.approx(
            fit.chi_squared_map.serial_trails_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_overscan_no_trails[
            0
        ] != pytest.approx(fit.chi_squared_map.serial_overscan_no_trails_frame, 1.0e-2)
