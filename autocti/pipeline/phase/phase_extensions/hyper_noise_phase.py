import copy

import autofit as af
from .hyper_phase import HyperPhase
import numpy as np
from autocti.charge_injection import ci_data, ci_fit, ci_hyper


class HyperNoisePhase(HyperPhase):
    def __init__(self, phase):

        super().__init__(phase=phase, hyper_name="hyper_noise")

    class Analysis(af.Analysis):
        def __init__(self, ci_datas_masked_full, model_images):
            """
            An analysis to fit the noise for a single galaxy image.
            Parameters
            ----------
            masked_imaging: LensData
                Lens instrument, including an image and noise
            hyper_noise_scaling_map: ndarray
                An image produce of the overall system by a model
            hyper_noise_image_1d_path_dict: ndarray
                The contribution of one galaxy to the model image
            """

            self.ci_datas_masked_full = ci_datas_masked_full
            self.model_images = model_images

            # self.plot_hyper_noise_subplot = af.conf.instance.visualize.get(
            #     "plots", "plot_hyper_noise_subplot", bool
            # )

        def visualize(self, instance, image_path, during_analysis):

            pass

        def fit(self, instance):
            """
            Fit the model image to the real image by scaling the hyper_noise noise.
            Parameters
            ----------
            instance: ModelInstance
                A model instance with a hyper_noise galaxy property
            Returns
            -------
            fit: float
            """

            hyper_noise_scalars = self.hyper_noise_scalars_from_instance(
                instance=instance
            )

            fits = list(
                map(
                    lambda data, model_image: self.fit_for_ci_data_model_image_and_hyper_noise_scalars(
                        ci_data=data,
                        model_image=model_image,
                        hyper_noise_scalars=hyper_noise_scalars,
                    ),
                    self.ci_datas_masked_full,
                    self.model_images,
                )
            )

            return np.sum(list(map(lambda fit: fit.figure_of_merit, fits)))

        def hyper_noise_scalars_from_instance(self, instance):
            return [
                instance.hyper_noise_scalar_of_ci_regions,
                instance.hyper_noise_scalar_of_parallel_trails,
                instance.hyper_noise_scalar_of_serial_trails,
                instance.hyper_noise_scalar_of_serial_overscan_above_trails,
            ]

        def fit_for_ci_data_model_image_and_hyper_noise_scalars(
            self, ci_data, model_image, hyper_noise_scalars
        ):

            hyper_noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(
                noise_scaling_maps=ci_data.noise_scaling_maps,
                hyper_noise_scalars=hyper_noise_scalars,
                noise_map=ci_data.noise_map,
            )

            return ci_fit.CIDataFit(
                ci_masked_imaging=ci_data,
                noise_map=hyper_noise_map,
                model_data=model_image,
            )

        @classmethod
        def describe(cls, instance):
            return "Running hyper_noise galaxy fit for HyperGalaxy:\n{}".format(
                instance.hyper_noise
            )

    def run_hyper(self, ci_datas, pool, results=None):
        """
        Run a fit for each galaxy from the previous phase.
        Parameters
        ----------
        simulator: LensData
        results: ResultsCollection
            Results from all previous phases
        Returns
        -------
        results: HyperGalaxyResults
            A collection of results, with one item per a galaxy
        """
        phase = self.make_hyper_phase()
        phase.hyper_noise_scalar_of_ci_regions = ci_hyper.CIHyperNoiseScalar
        phase.hyper_noise_scalar_of_parallel_trails = ci_hyper.CIHyperNoiseScalar
        phase.hyper_noise_scalar_of_serial_trails = ci_hyper.CIHyperNoiseScalar
        phase.hyper_noise_scalar_of_serial_overscan_above_trails = (
            ci_hyper.CIHyperNoiseScalar
        )

        masks = phase.masks_for_analysis_from_ci_datas(ci_datas=ci_datas)

        noise_scaling_maps = phase.noise_scaling_maps_from_total_images_and_results(
            total_images=len(ci_datas), results=results
        )

        ci_datas_masked_full = list(
            map(
                lambda data, mask, maps: ci_data.CIMaskedImaging(
                    image=data.image,
                    noise_map=data.noise_map,
                    ci_pre_cti=data.ci_pre_cti,
                    mask=mask,
                    ci_pattern=data.ci_pattern,
                    ci_frame=data.ci_frame,
                    noise_scaling_maps=maps,
                ),
                ci_datas,
                masks,
                noise_scaling_maps,
            )
        )

        model_images = list(
            map(
                lambda most_likely_fit: most_likely_fit.model_image,
                results.last.most_likely_full_fits,
            )
        )

        hyper_result = copy.deepcopy(results.last)
        hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
            hyper_result.constant
        )

        hyper_result.analysis.uses_hyper_images = True
        hyper_result.analysis.model_images = model_images

        phase.optimizer.variable.parallel_species = []
        phase.optimizer.variable.parallel_ccd_volume = []
        phase.optimizer.phase_tag = ""

        phase.optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
            "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
        )
        phase.optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
            "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
        )
        phase.optimizer.n_live_points = af.conf.instance.non_linear.get(
            "MultiNest", "extension_hyper_galaxy_n_live_points", int
        )
        phase.optimizer.multimodal = af.conf.instance.non_linear.get(
            "MultiNest", "extension_hyper_galaxy_multimodal", bool
        )

        analysis = self.Analysis(
            ci_datas_masked_full=ci_datas_masked_full, model_images=model_images
        )

        result = phase.optimizer.fit(analysis)

        def transfer_field(name):
            if hasattr(result.constant, name):
                setattr(hyper_result.constant, name, getattr(result.constant, name))
                setattr(hyper_result.variable, name, getattr(result.variable, name))

        transfer_field(name="hyper_noise_scalar_of_ci_regions")
        transfer_field(name="hyper_noise_scalar_of_parallel_trails")
        transfer_field(name="hyper_noise_scalar_of_serial_trails")
        transfer_field(name="hyper_noise_scalar_of_serial_overscan_above_trails")

        return hyper_result
