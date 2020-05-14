from os import path

import arctic as ac
import autocti.charge_injection as ci
import autofit as af
import pytest
from autocti import exc
from autocti.pipeline.phase.ci_imaging import PhaseCIImaging
from test_autocti.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestChecks:
    def test__parallel_and_serial_checks_raise_exception(
        self, phase_ci_imaging_7x7, ci_imaging_7x7
    ):

        phase_ci_imaging_7x7.meta_dataset.parallel_total_density_range = (1.0, 2.0)

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = af.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]

        analysis.check_total_density_within_range(instance=instance)

        instance = af.ModelInstance()
        instance.parallel_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)

        phase_ci_imaging_7x7.meta_dataset.parallel_total_density_range = None
        phase_ci_imaging_7x7.meta_dataset.serial_total_density_range = (1.0, 2.0)

        analysis = phase_ci_imaging_7x7.make_analysis(
            datasets=[ci_imaging_7x7], clocker=None
        )

        instance = af.ModelInstance()
        instance.serial_traps = [ac.Trap(density=0.75), ac.Trap(density=0.75)]

        analysis.check_total_density_within_range(instance=instance)

        instance = af.ModelInstance()
        instance.serial_traps = [ac.Trap(density=1.1), ac.Trap(density=1.1)]

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(instance=instance)


class TestFit:
    def test__log_likelihood_via_analysis__matches_manual_fit(
        self, ci_imaging_7x7, ci_pre_cti_7x7, traps_x1, ccd_volume, parallel_clocker
    ):

        phase = PhaseCIImaging(
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7], clocker=parallel_clocker
        )
        instance = phase.model.instance_from_unit_vector([])

        log_likelihood_via_analysis = analysis.log_likelihood_function(
            instance=instance
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=ci_pre_cti_7x7,
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=analysis.masked_ci_imagings[0], ci_post_cti=ci_post_cti
        )

        assert fit.log_likelihood == log_likelihood_via_analysis

    def test__extracted_fits_from_instance_and_ci_imaging(
        self, ci_imaging_7x7, mask_7x7, traps_x1, ccd_volume, parallel_clocker
    ):

        phase = PhaseCIImaging(
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
            non_linear_class=mock_pipeline.MockNLO,
            columns=(0, 1),
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7], clocker=parallel_clocker
        )
        instance = phase.model.instance_from_unit_vector([])

        fits = analysis.fits_from_instance(instance=instance)

        mask_from_phase = phase.meta_dataset.mask_for_analysis_from_dataset(
            dataset=ci_imaging_7x7, mask=mask_7x7
        )

        masked_ci_imaging = ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7, mask=mask_from_phase, parallel_columns=(0, 1)
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=masked_ci_imaging.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging, ci_post_cti=ci_post_cti
        )

        assert fits[0].image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood)

    def test__full_fits_from_instance_and_ci_imaging(
        self, ci_imaging_7x7, mask_7x7, traps_x1, ccd_volume, parallel_clocker
    ):

        phase = PhaseCIImaging(
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
            non_linear_class=mock_pipeline.MockNLO,
            columns=(0, 1),
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7], clocker=parallel_clocker
        )
        instance = phase.model.instance_from_unit_vector([])

        fits = analysis.fits_full_dataset_from_instance(instance=instance)

        ci_post_cti = parallel_clocker.add_cti(
            image=ci_imaging_7x7.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
        )

        fit = ci.CIFitImaging(masked_ci_imaging=ci_imaging_7x7, ci_post_cti=ci_post_cti)

        assert fits[0].image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood)

    def test__extracted_fits_from_instance_and_ci_imaging__include_noise_scaling(
        self,
        ci_imaging_7x7,
        mask_7x7,
        traps_x1,
        ccd_volume,
        parallel_clocker,
        ci_pattern_7x7,
    ):

        phase = PhaseCIImaging(
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
            hyper_noise_scalar_of_ci_regions=ci.CIHyperNoiseScalar,
            non_linear_class=mock_pipeline.MockNLO,
            columns=(0, 1),
            phase_name="test_phase",
        )

        noise_scaling_maps_list_of_ci_regions = [
            ci.CIFrame.ones(
                shape_2d=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions
            ),
        )
        instance = phase.model.instance_from_prior_medians()

        fits = analysis.fits_from_instance(instance=instance, hyper_noise_scale=True)

        mask = ci.CIMask.unmasked(
            shape_2d=ci_imaging_7x7.shape_2d, pixel_scales=ci_imaging_7x7.pixel_scales
        )

        masked_ci_imaging_7x7 = ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
            parallel_columns=(0, 1),
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=masked_ci_imaging_7x7.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise_scalar_of_ci_regions],
        )

        assert fits[0].image.shape == (7, 1)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)

    def test__full_fits_from_instance_and_ci_imaging__include_noise_scaling(
        self,
        ci_imaging_7x7,
        mask_7x7,
        traps_x1,
        ccd_volume,
        parallel_clocker,
        ci_pattern_7x7,
    ):

        phase = PhaseCIImaging(
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
            hyper_noise_scalar_of_ci_regions=ci.CIHyperNoiseScalar,
            non_linear_class=mock_pipeline.MockNLO,
            columns=(0, 1),
            phase_name="test_phase",
        )

        noise_scaling_maps_list_of_ci_regions = [
            ci.CIFrame.ones(
                shape_2d=(7, 7), pixel_scales=1.0, ci_pattern=ci_pattern_7x7
            )
        ]

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
                noise_scaling_maps_list_of_ci_regions=noise_scaling_maps_list_of_ci_regions
            ),
        )
        instance = phase.model.instance_from_prior_medians()

        fits = analysis.fits_full_dataset_from_instance(
            instance=instance, hyper_noise_scale=True
        )

        ci_post_cti = parallel_clocker.add_cti(
            image=ci_imaging_7x7.ci_pre_cti,
            parallel_traps=traps_x1,
            parallel_ccd_volume=ccd_volume,
        )

        mask = ci.CIMask.unmasked(
            shape_2d=ci_imaging_7x7.shape_2d, pixel_scales=ci_imaging_7x7.pixel_scales
        )

        masked_ci_imaging_7x7 = ci.MaskedCIImaging(
            ci_imaging=ci_imaging_7x7,
            mask=mask,
            noise_scaling_maps=[noise_scaling_maps_list_of_ci_regions[0]],
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[instance.hyper_noise_scalar_of_ci_regions],
        )

        assert fits[0].image.shape == (7, 7)
        assert fit.log_likelihood == pytest.approx(fits[0].log_likelihood, 1.0e-4)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=ci_post_cti,
            hyper_noise_scalars=[ci.CIHyperNoiseScalar(scale_factor=0.0)],
        )

        assert fit.log_likelihood != pytest.approx(fits[0].log_likelihood, 1.0e-4)

    def test__hyper_noise_scalar_properties_of_phase(
        self, ci_imaging_7x7, ci_pattern_7x7, parallel_clocker
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

        phase = PhaseCIImaging(
            phase_name="test_phase",
            hyper_noise_scalar_of_ci_regions=ci.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ci.CIHyperNoiseScalar,
        )

        analysis = phase.make_analysis(
            datasets=[ci_imaging_7x7],
            clocker=parallel_clocker,
            results=mock_pipeline.MockResults(
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

        assert instance.hyper_noise_scalar_of_ci_regions == 0.5
        assert instance.hyper_noise_scalar_of_parallel_trails == 0.5
