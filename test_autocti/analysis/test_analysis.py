from os import path

import autofit as af
import autocti as ac
import pytest
from autocti import exc
from autocti.pipeline.phase.ci_imaging.phase import PhaseCIImaging
from autocti.mock import mock


class TestAnalysis:
    def test__parallel_and_serial_checks_raise_exception(self, ci_imaging_7x7):

        analysis = ac.AnalysisCIImaging(
            ci_imagings=[ci_imaging_7x7],
            clocker=None,
            settings_cti=ac.SettingsCTI(parallel_total_density_range=(1.0, 2.0)),
        )

        instance = af.ModelInstance()
        instance.parallel_traps = [
            ac.TrapInstantCapture(density=1.1),
            ac.TrapInstantCapture(density=1.1),
        ]
        instance.serial_traps = []

        with pytest.raises(exc.PriorException):
            analysis.log_likelihood_function(instance=instance)

        instance = af.ModelInstance()
        instance.parallel_traps = []
        instance.serial_traps = [
            ac.TrapInstantCapture(density=1.1),
            ac.TrapInstantCapture(density=1.1),
        ]

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
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions
            ),
        )
        instance = phase.model.instance_from_prior_medians()

        fits = analysis.fits_from_instance(instance=instance, hyper_noise_scale=True)

        mask = ac.ci.CIMask2D.unmasked(
            shape_native=ci_imaging_7x7.shape_native,
            pixel_scales=ci_imaging_7x7.pixel_scales,
        )

        masked_ci_imaging_7x7 = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
            settings=ac.ci.SettingsMaskedCIImaging(parallel_columns=(0, 1)),
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=masked_ci_imaging_7x7.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise_scalar_of_ci_regions],
        )

        assert fits[0].image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ac.ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)

    def test__full_fits_from_instance_and_ci_imaging__include_noise_scaling(
        self,
        ci_imaging_7x7,
        mask_7x7_unmasked,
        traps_x1,
        ccd,
        parallel_clocker,
        ci_pattern_7x7,
    ):

        phase = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            parallel_traps=traps_x1,
            parallel_ccd=ccd,
            hyper_noise_scalar_of_ci_regions=ac.ci.CIHyperNoiseScalar,
            settings=ac.SettingsPhaseCIImaging(
                settings_masked_ci_imaging=ac.ci.SettingsMaskedCIImaging(
                    parallel_columns=(0, 1)
                )
            ),
        )

        noise_scaling_maps_list_of_ci_regions = [
            ac.ci.CIFrame.ones(
                shape_native=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions
            ),
        )
        instance = phase.model.instance_from_prior_medians()

        fits = analysis.fits_full_dataset_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=ci_imaging_7x7.ci_pre_cti, parallel_traps=traps_x1, parallel_ccd=ccd
        )

        mask = ac.ci.CIMask2D.unmasked(
            shape_native=ci_imaging_7x7.shape_native,
            pixel_scales=ci_imaging_7x7.pixel_scales,
        )

        masked_ci_imaging_7x7 = ac.ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
        )

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise_scalar_of_ci_regions],
        )

        assert fits[0].image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ac.ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ac.ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)

    def test__hyper_noise_scalar_properties_of_phase(
        self, ci_imaging_7x7, ci_pattern_7x7, parallel_clocker
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

        phase = PhaseCIImaging(
            search=mock.MockSearch(name="test_phase"),
            hyper_noise_scalar_of_ci_regions=ac.ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ac.ci.CIHyperNoiseScalar,
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions,
                noise_scaling_maps_list_of_parallel_trails=noise_scaling_maps_list_of_parallel_trails,
            ),
        )

        instance = phase.model.instance_from_prior_medians()

        hyper_noise_scalars = analysis.hyper_noise_scalars_from_instance(
            instance=instance
        )

        assert len(hyper_noise_scalars) == 2
        assert len(phase.model.priors) == 2

        assert instance.hyper_noise_scalar_of_ci_regions == 5.0
        assert instance.hyper_noise_scalar_of_parallel_trails == 5.0
