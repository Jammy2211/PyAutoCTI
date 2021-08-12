import autofit as af
import autocti as ac
import pytest
from autocti import exc
from autocti.mock import mock
from autocti.analysis import result as res


class TestAnalysis:
    def test__parallel_and_serial_checks_raise_exception(self, imaging_ci_7x7):

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI2D,
                parallel_traps=[
                    ac.TrapInstantCapture(density=1.1),
                    ac.TrapInstantCapture(density=1.1),
                ],
                parallel_ccd=ac.CCDPhase(),
            )
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=imaging_ci_7x7,
            clocker=None,
            settings_cti=ac.SettingsCTI2D(parallel_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI2D,
                serial_traps=[
                    ac.TrapInstantCapture(density=1.1),
                    ac.TrapInstantCapture(density=1.1),
                ],
                serial_ccd=ac.CCDPhase(),
            )
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=[imaging_ci_7x7],
            clocker=None,
            settings_cti=ac.SettingsCTI2D(serial_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)


class TestAnalysisDatasetLine:
    def test__make_result__result_line_is_returned(
        self, dataset_line_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=dataset_line_7, clocker=clocker_1d
        )

        search = mock.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultDatasetLine)

    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, dataset_line_7, pre_cti_data_7, traps_x1, ccd, clocker_1d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=dataset_line_7, clocker=clocker_1d
        )

        instance = model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        post_cti_data = clocker_1d.add_cti(
            pre_cti_data=pre_cti_data_7.native, traps=traps_x1, ccd=ccd
        )

        fit = ac.FitDatasetLine(
            dataset_line=analysis.dataset_line, post_cti_data=post_cti_data
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__extracted_fits_from_instance_and_line_ci(
        self, dataset_line_7, mask_1d_7_unmasked, traps_x1, ccd, clocker_1d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI1D, traps=traps_x1, ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        masked_line_ci = dataset_line_7.apply_mask(mask=mask_1d_7_unmasked)

        post_cti_data = clocker_1d.add_cti(
            pre_cti_data=masked_line_ci.pre_cti_data, traps=traps_x1, ccd=ccd
        )

        analysis = ac.AnalysisDatasetLine(
            dataset_line=masked_line_ci, clocker=clocker_1d
        )

        instance = model.instance_from_unit_vector([])

        fit_analysis = analysis.fit_from_instance(instance=instance)

        fit = ac.FitDatasetLine(
            dataset_line=masked_line_ci, post_cti_data=post_cti_data
        )

        assert fit.dataset.data.shape == (7,)
        assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)


class TestAnalysisImagingCI:
    def test__make_result__result_imaging_is_returned(
        self, imaging_ci_7x7, pre_cti_data_7x7, traps_x1, ccd, parallel_clocker_2d
    ):
        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI2D, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=imaging_ci_7x7, clocker=parallel_clocker_2d
        )

        search = mock.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImagingCI)

    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, imaging_ci_7x7, pre_cti_data_7x7, traps_x1, ccd, parallel_clocker_2d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI2D, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=imaging_ci_7x7, clocker=parallel_clocker_2d
        )

        instance = model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        post_cti_data = parallel_clocker_2d.add_cti(
            pre_cti_data=pre_cti_data_7x7.native,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.FitImagingCI(
            imaging=analysis.dataset_ci, post_cti_data=post_cti_data
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__full_and_extracted_fits_from_instance_and_imaging_ci(
        self, imaging_ci_7x7, mask_2d_7x7_unmasked, traps_x1, ccd, parallel_clocker_2d
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI2D, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsImagingCI(parallel_columns=(0, 1))
        )

        post_cti_data = parallel_clocker_2d.add_cti(
            pre_cti_data=masked_imaging_ci.pre_cti_data,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=masked_imaging_ci, clocker=parallel_clocker_2d
        )

        instance = model.instance_from_unit_vector([])

        fit_analysis = analysis.fit_from_instance(instance=instance)

        fit = ac.ci.FitImagingCI(imaging=masked_imaging_ci, post_cti_data=post_cti_data)

        assert fit.image.shape == (7, 1)
        assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)

        fit_full_analysis = analysis.fit_full_dataset_from_instance(instance=instance)

        fit = ac.ci.FitImagingCI(imaging=imaging_ci_7x7, post_cti_data=post_cti_data)

        assert fit.image.shape == (7, 7)
        assert fit_full_analysis.log_likelihood == pytest.approx(fit.log_likelihood)

    def test__extracted_fits_from_instance_and_imaging_ci__include_noise_scaling(
        self,
        imaging_ci_7x7,
        mask_2d_7x7_unmasked,
        traps_x1,
        ccd,
        parallel_clocker_2d,
        layout_ci_7x7,
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI2D, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(
                ac.ci.HyperCINoiseCollection, regions_ci=ac.ci.HyperCINoiseScalar
            ),
        )

        noise_scaling_map_list_list_of_regions_ci = [
            ac.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0)
        ]

        imaging_ci_7x7.noise_scaling_map_list = [
            noise_scaling_map_list_list_of_regions_ci[0]
        ]

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsImagingCI(parallel_columns=(0, 1))
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=masked_imaging_ci, clocker=parallel_clocker_2d
        )

        instance = model.instance_from_prior_medians()

        fit_analysis = analysis.fit_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        post_cti_data = parallel_clocker_2d.add_cti(
            pre_cti_data=masked_imaging_ci.pre_cti_data,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=[instance.hyper_noise.regions_ci],
        )

        assert fit.image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=[ac.ci.HyperCINoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

        fit_full_analysis = analysis.fit_full_dataset_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=[instance.hyper_noise.regions_ci],
        )

        assert fit.image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(
            fit_full_analysis.log_likelihood, 1.0e-4
        )

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalars=[ac.ci.HyperCINoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(
            fit_full_analysis.log_likelihood, 1.0e-4
        )
