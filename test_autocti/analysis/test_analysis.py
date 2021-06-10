from os import path

import autofit as af
import autocti as ac
import pytest
from autocti import exc


class TestAnalysis:
    def test__parallel_and_serial_checks_raise_exception(self, imaging_ci_7x7):

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI,
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
            settings_cti=ac.SettingsCTI(parallel_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI,
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
            settings_cti=ac.SettingsCTI(serial_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)


class TestAnalysisImagingCI:
    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, imaging_ci_7x7, pre_cti_image_7x7, traps_x1, ccd, parallel_clocker
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=imaging_ci_7x7, clocker=parallel_clocker
        )

        instance = model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        post_cti_image = parallel_clocker.add_cti(
            image_pre_cti=pre_cti_image_7x7.native,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.FitImagingCI(
            imaging=analysis.dataset_ci, post_cti_image=post_cti_image
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__full_and_extracted_fits_from_instance_and_imaging_ci(
        self, imaging_ci_7x7, mask_2d_7x7_unmasked, traps_x1, ccd, parallel_clocker
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.HyperCINoiseCollection),
        )

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)
        masked_imaging_ci = masked_imaging_ci.apply_settings(
            settings=ac.ci.SettingsImagingCI(parallel_columns=(0, 1))
        )

        post_cti_image = parallel_clocker.add_cti(
            image_pre_cti=masked_imaging_ci.pre_cti_image,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        analysis = ac.AnalysisImagingCI(
            dataset_ci=masked_imaging_ci, clocker=parallel_clocker
        )

        instance = model.instance_from_unit_vector([])

        fit_analysis = analysis.fit_from_instance(instance=instance)

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci, post_cti_image=post_cti_image
        )

        assert fit.image.shape == (7, 1)
        assert fit_analysis.log_likelihood == pytest.approx(fit.log_likelihood)

        fit_full_analysis = analysis.fit_full_dataset_from_instance(instance=instance)

        fit = ac.ci.FitImagingCI(imaging=imaging_ci_7x7, post_cti_image=post_cti_image)

        assert fit.image.shape == (7, 7)
        assert fit_full_analysis.log_likelihood == pytest.approx(fit.log_likelihood)

    def test__extracted_fits_from_instance_and_imaging_ci__include_noise_scaling(
        self,
        imaging_ci_7x7,
        mask_2d_7x7_unmasked,
        traps_x1,
        ccd,
        parallel_clocker,
        layout_ci_7x7,
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
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
            dataset_ci=masked_imaging_ci, clocker=parallel_clocker
        )

        instance = model.instance_from_prior_medians()

        fit_analysis = analysis.fit_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        post_cti_image = parallel_clocker.add_cti(
            image_pre_cti=masked_imaging_ci.pre_cti_image,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=[instance.hyper_noise.regions_ci],
        )

        assert fit.image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=[ac.ci.HyperCINoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fit_analysis.log_likelihood, 1.0e-4)

        fit_full_analysis = analysis.fit_full_dataset_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        masked_imaging_ci = imaging_ci_7x7.apply_mask(mask=mask_2d_7x7_unmasked)

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=[instance.hyper_noise.regions_ci],
        )

        assert fit.image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(
            fit_full_analysis.log_likelihood, 1.0e-4
        )

        fit = ac.ci.FitImagingCI(
            imaging=masked_imaging_ci,
            post_cti_image=post_cti_image,
            hyper_noise_scalars=[ac.ci.HyperCINoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(
            fit_full_analysis.log_likelihood, 1.0e-4
        )
