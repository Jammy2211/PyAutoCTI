from os import path

import autofit as af
import autocti as ac
import pytest
from autocti import exc


class TestAnalysis:
    def test__parallel_and_serial_checks_raise_exception(self, ci_imaging_7x7):

        model = af.CollectionPriorModel(
            cti=af.Model(
                ac.CTI,
                parallel_traps=[
                    ac.TrapInstantCapture(density=1.1),
                    ac.TrapInstantCapture(density=1.1),
                ],
                parallel_ccd=ac.CCD(),
            )
        )

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[ci_imaging_7x7],
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
                serial_ccd=ac.CCD(),
            )
        )

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[ci_imaging_7x7],
            clocker=None,
            settings_cti=ac.SettingsCTI(serial_total_density_range=(1.0, 2.0)),
        )

        instance = model.instance_from_prior_medians()

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)


class TestAnalysisCIImaging:
    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, ci_imaging_7x7, ci_pre_cti_7x7, traps_x1, ccd, parallel_clocker
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.CIHyperNoiseCollection),
        )

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[ci_imaging_7x7], clocker=parallel_clocker
        )

        instance = model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=ci_pre_cti_7x7, parallel_traps=traps_x1, parallel_ccd=ccd
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=analysis.ci_imagings[0], ci_post_cti=ci_post_cti
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__full_and_extracted_fits_from_instance_and_ci_imaging(
        self, ci_imaging_7x7, mask_7x7_unmasked, traps_x1, ccd, parallel_clocker
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(ac.ci.CIHyperNoiseCollection),
        )

        masked_ci_imaging = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask_7x7_unmasked,
            settings=ac.ci.SettingsMaskedCIImaging(parallel_columns=(0, 1)),
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=masked_ci_imaging.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[masked_ci_imaging], clocker=parallel_clocker
        )

        instance = model.instance_from_unit_vector([])

        fits = analysis.fits_from_instance(instance=instance)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging, ci_post_cti=ci_post_cti
        )

        assert fits[0].image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood)

        fits = analysis.fits_full_dataset_from_instance(instance=instance)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=ci_imaging_7x7, ci_post_cti=ci_post_cti
        )

        assert fits[0].image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood)

    def test__extracted_fits_from_instance_and_ci_imaging__include_noise_scaling(
        self,
        ci_imaging_7x7,
        mask_7x7_unmasked,
        traps_x1,
        ccd,
        parallel_clocker,
        ci_pattern_7x7,
    ):

        model = af.CollectionPriorModel(
            cti=af.Model(ac.CTI, parallel_traps=traps_x1, parallel_ccd=ccd),
            hyper_noise=af.Model(
                ac.ci.CIHyperNoiseCollection, ci_regions=ac.ci.CIHyperNoiseScalar
            ),
        )

        noise_scaling_maps_list_of_ci_regions = [
            ac.ci.CIFrame.ones(
                shape_native=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]

        masked_ci_imaging = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask_7x7_unmasked,
            settings=ac.ci.SettingsMaskedCIImaging(parallel_columns=(0, 1)),
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
        )

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[masked_ci_imaging], clocker=parallel_clocker
        )

        instance = model.instance_from_prior_medians()

        fits = analysis.fits_from_instance(instance=instance, hyper_noise_scale=True)

        ci_post_cti = parallel_clocker.add_cti(
            image=masked_ci_imaging.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise.ci_regions],
        )

        assert fits[0].image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ac.ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fits = analysis.fits_full_dataset_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        masked_ci_imaging = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask_7x7_unmasked,
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise.ci_regions],
        )

        assert fits[0].image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ac.ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)
