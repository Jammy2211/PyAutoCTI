from os import path

import autocti as ac
import numpy as np
import pytest
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from autocti.pipeline.phase.dataset.result import Result
from autocti.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__fits_to_extracted_and_full_datasets_available(
        self, ci_imaging_7x7, parallel_clocker, samples_with_result
    ):

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase_2", samples=samples_with_result),
            settings=ac.SettingsPhaseCIImaging(
                settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                    parallel_columns=(0, 1)
                )
            ),
        )

        result = phase_ci_imaging_7x7.run(
            dataset_list=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(),
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
        self, ci_imaging_7x7, parallel_clocker, ci_pattern_7x7, samples_with_result
    ):

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

        phase_ci_imaging_7x7 = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase_2", samples=samples_with_result),
            hyper_noise_scalar_of_ci_regions=ac.ci.CIHyperNoiseScalar(),
            hyper_noise_scalar_of_parallel_trails=ac.ci.CIHyperNoiseScalar(),
            hyper_noise_scalar_of_serial_trails=ac.ci.CIHyperNoiseScalar(),
            hyper_noise_scalar_of_serial_overscan_no_trails=ac.ci.CIHyperNoiseScalar(),
        )

        result = phase_ci_imaging_7x7.run(
            dataset_list=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
                noise_scaling_maps_list_of_serial_trails=noise_scaling_maps_list_of_serial_trails,
                noise_scaling_maps_list_of_serial_overscan_no_trails=noise_scaling_maps_list_of_serial_overscan_no_trails,
            ),
        )

        ci_post_cti = parallel_clocker.add_cti(image=ci_imaging_7x7.ci_pre_cti)

        mask = ac.ci.CIMask.unmasked(
            shape_native=ci_imaging_7x7.shape_native,
            pixel_scales=ci_imaging_7x7.pixel_scales,
        )
        masked_ci_imaging_7x7 = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[
                noise_scaling_maps_list_of_ci_regions[0],
                noise_scaling_maps_list_of_parallel_trails[0],
                noise_scaling_maps_list_of_serial_trails[0],
                noise_scaling_maps_list_of_serial_overscan_no_trails[0],
            ],
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[
                ac.ci.CIHyperNoiseScalar(scale_factor=0.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=0.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=0.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=0.0),
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

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[
                ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
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
