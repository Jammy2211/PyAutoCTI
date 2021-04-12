import autofit as af
import autocti as ac
from autocti.analysis import result as res

from os import path
import numpy as np
import pytest


class TestResult:
    def test__result_contains_instance_with_cti_model(
        self, analysis_imaging_ci_7x7, samples_with_result
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=None,
            search=None,
        )

        assert isinstance(result.instance.cti.parallel_traps[0], ac.TrapInstantCapture)
        assert isinstance(result.instance.cti.parallel_ccd, ac.CCD)

    def test__clocker_passed_as_result_correctly(
        self, analysis_imaging_ci_7x7, samples_with_result, parallel_clocker
    ):

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=None,
            search=None,
        )

        assert isinstance(result.clocker, ac.Clocker)
        assert result.clocker.parallel_express == parallel_clocker.parallel_express


class TestResultDataset:
    def test__masks_available_as_property(
        self,
        analysis_imaging_ci_7x7,
        samples_with_result,
        parallel_clocker,
        traps_x1,
        ccd,
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd)
        )
        result = res.ResultDataset(
            samples=samples_with_result,
            analysis=analysis_imaging_ci_7x7,
            model=model,
            search=None,
        )

        assert (result.masks[0] == np.full(fill_value=False, shape=(7, 7))).all()


class TestResultCIImaging:
    def test__fits_to_extracted_and_full_datasets_available(
        self, imaging_ci_7x7, mask_7x7_unmasked, parallel_clocker, samples_with_result
    ):

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_7x7_unmasked)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsCIImaging(parallel_columns=(0, 1))
        )

        analysis = ac.AnalysisCIImaging(
            dataset_list=[masked_imaging_ci], clocker=parallel_clocker
        )

        result = res.ResultCIImaging(
            samples=samples_with_result, analysis=analysis, model=None, search=None
        )

        assert (
            result.max_log_likelihood_fits[0].mask
            == np.full(fill_value=False, shape=(7, 1))
        ).all()

        assert (
            result.max_log_likelihood_full_fits[0].mask
            == np.full(fill_value=False, shape=(7, 7))
        ).all()

    def test__noise_scaling_maps_list_of_result__are_correct(
        self,
        imaging_ci_7x7,
        mask_7x7_unmasked,
        parallel_clocker,
        pattern_ci_7x7,
        samples_with_result,
        traps_x1,
        ccd,
    ):

        noise_scaling_maps_list_of_ci_regions = [
            ac.ci.CIFrame.ones(
                shape_native=(7, 7), pixel_scales=1.0, pattern_ci=pattern_ci_7x7
            )
        ]
        noise_scaling_maps_list_of_parallel_trails = [
            ac.ci.CIFrame.full(
                fill_value=2.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                pattern_ci=pattern_ci_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_trails = [
            ac.ci.CIFrame.full(
                fill_value=3.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                pattern_ci=pattern_ci_7x7,
            )
        ]
        noise_scaling_maps_list_of_serial_overscan_no_trails = [
            ac.ci.CIFrame.full(
                fill_value=4.0,
                shape_native=(7, 7),
                pixel_scales=1.0,
                pattern_ci=pattern_ci_7x7,
            )
        ]

        imaging_ci_7x7.noise_scaling_maps = [
            noise_scaling_maps_list_of_ci_regions[0],
            noise_scaling_maps_list_of_parallel_trails[0],
            noise_scaling_maps_list_of_serial_trails[0],
            noise_scaling_maps_list_of_serial_overscan_no_trails[0],
        ]

        masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_7x7_unmasked)

        analysis = ac.AnalysisCIImaging(
            dataset_list=[masked_imaging_ci_7x7], clocker=parallel_clocker
        )

        fit_list = analysis.fits_from_instance(
            instance=samples_with_result.max_log_likelihood_instance
        )

        result = res.ResultCIImaging(
            samples=samples_with_result, analysis=analysis, model=None, search=None
        )

        assert result.noise_scaling_maps_list_of_ci_regions[0] == pytest.approx(
            fit_list[0].chi_squared_map.ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_parallel_trails[0] == pytest.approx(
            fit_list[0].chi_squared_map.parallel_non_ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_trails[0] == pytest.approx(
            fit_list[0].chi_squared_map.serial_trails_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_overscan_no_trails[
            0
        ] == pytest.approx(
            fit_list[0].chi_squared_map.serial_overscan_no_trails_frame, 1.0e-2
        )

        assert result.noise_scaling_maps_list_of_ci_regions[0][1, 1] == pytest.approx(
            16.25, 1.0e-1
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

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI,
                parallel_traps=traps_x1,
                parallel_ccd=ccd,
                serial_traps=traps_x1,
                serial_ccd=ccd,
            ),
            hyper_noise=af.Model(
                ac.ci.CIHyperNoiseCollection,
                ci_regions=ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                parallel_trails=ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                serial_trails=ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
                serial_overscan_no_trails=ac.ci.CIHyperNoiseScalar(scale_factor=1.0),
            ),
        )

        instance = model.instance_from_prior_medians()

        fit_list = analysis.fits_from_instance(instance=instance)

        assert result.noise_scaling_maps_list_of_ci_regions[0] != pytest.approx(
            fit_list[0].chi_squared_map.ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_parallel_trails[0] != pytest.approx(
            fit_list[0].chi_squared_map.parallel_non_ci_regions_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_trails[0] != pytest.approx(
            fit_list[0].chi_squared_map.serial_trails_frame, 1.0e-2
        )
        assert result.noise_scaling_maps_list_of_serial_overscan_no_trails[
            0
        ] != pytest.approx(
            fit_list[0].chi_squared_map.serial_overscan_no_trails_frame, 1.0e-2
        )

    def test__noise_scaling_maps_are_setup_correctly(
        self,
        imaging_ci_7x7,
        mask_7x7_unmasked,
        pattern_ci_7x7,
        parallel_clocker,
        samples_with_result,
    ):

        imaging_ci_7x7.cosmic_ray_map = None

        masked_imaging_ci_7x7 = imaging_ci_7x7.apply_mask(mask=mask_7x7_unmasked)

        analysis = ac.AnalysisCIImaging(
            dataset_list=[masked_imaging_ci_7x7, masked_imaging_ci_7x7],
            clocker=parallel_clocker,
        )

        result = res.ResultCIImaging(
            samples=samples_with_result, analysis=analysis, model=None, search=None
        )

        assert (
            result.noise_scaling_maps_list[0][0]
            == result.noise_scaling_maps_list_of_ci_regions[0]
        ).all()

        assert (
            result.noise_scaling_maps_list[0][1]
            == result.noise_scaling_maps_list_of_parallel_trails[0]
        ).all()

        assert (
            result.noise_scaling_maps_list[0][2]
            == result.noise_scaling_maps_list_of_serial_trails[0]
        ).all()

        assert (
            result.noise_scaling_maps_list[0][3]
            == result.noise_scaling_maps_list_of_serial_overscan_no_trails[0]
        ).all()

        assert (
            result.noise_scaling_maps_list[1][0]
            == result.noise_scaling_maps_list_of_ci_regions[1]
        ).all()

        assert (
            result.noise_scaling_maps_list[1][1]
            == result.noise_scaling_maps_list_of_parallel_trails[1]
        ).all()

        assert (
            result.noise_scaling_maps_list[1][2]
            == result.noise_scaling_maps_list_of_serial_trails[1]
        ).all()

        assert (
            result.noise_scaling_maps_list[1][3]
            == result.noise_scaling_maps_list_of_serial_overscan_no_trails[1]
        ).all()
