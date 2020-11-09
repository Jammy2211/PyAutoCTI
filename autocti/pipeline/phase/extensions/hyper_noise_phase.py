import copy

import numpy as np
from autoconf import conf
from autocti.charge_injection import ci_imaging, ci_fit, ci_hyper
from autofit.non_linear import abstract_search

from .hyper_phase import HyperPhase


class HyperNoisePhase(HyperPhase):
    def __init__(self, phase):

        super().__init__(phase=phase, hyper_name="hyper_noise")

    class Analysis(abstract_search.Analysis):
        def __init__(self, masked_ci_dataset_full, model_images):
            """
            An analysis to fit the noise for a single galaxy image.
            Parameters
            ----------
            masked_imaging: LensData
                Lens instrument, including an image and noise
            hyper_noise_scaling_map: np.ndarray
                An image produce of the overall system by a model
            hyper_noise_image_1d_path_dict: np.ndarray
                The contribution of one galaxy to the model image
            """

            self.masked_ci_dataset_full = masked_ci_dataset_full
            self.model_images = model_images

            # self.plot_hyper_noise_subplot = conf.instance.visualize.get(
            #     "plots", "plot_hyper_noise_subplot", bool
            # )

        def visualize(self, paths, instance, during_analysis):

            pass

        def log_likelihood_function(self, instance):
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
                    self.masked_ci_dataset_full,
                    self.model_images,
                )
            )

            return np.sum(list(map(lambda fit: fit.figure_of_merit, fits)))

        def hyper_noise_scalars_from_instance(self, instance):
            return [
                instance.hyper_noise_scalar_of_ci_regions,
                instance.hyper_noise_scalar_of_parallel_trails,
                instance.hyper_noise_scalar_of_serial_trails,
                instance.hyper_noise_scalar_of_serial_overscan_no_trails,
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
                masked_ci_imaging=ci_data,
                noise_map=hyper_noise_map,
                model_data=model_image,
            )

        @classmethod
        def describe(cls, instance):
            return "Running hyper_noise galaxy fit for HyperGalaxy:\n{}".format(
                instance.hyper_noise
            )

    def run_hyper(self, datasets, pool, results=None):
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
        phase.hyper_noise_scalar_of_serial_overscan_no_trails = (
            ci_hyper.CIHyperNoiseScalar
        )

        masks = phase.mask_for_analysis_from_dataset(dataset=datasets)

        noise_scaling_maps_list = phase.noise_scaling_maps_list_from_total_images_and_results(
            total_images=len(datasets), results=results
        )

        masked_ci_dataset_full = list(
            map(
                lambda data, mask, maps: ci_imaging.MaskedCIImaging(
                    image=data.image,
                    noise_map=data.noise_map,
                    ci_pre_cti=data.ci_pre_cti,
                    mask=mask,
                    ci_pattern=data.ci_pattern,
                    ci_frame=data.ci_frame,
                    noise_scaling_maps_list=maps,
                ),
                datasets,
                masks,
                noise_scaling_maps_list,
            )
        )

        model_images = list(
            map(
                lambda max_log_likelihood_fit: max_log_likelihood_fit.model_image,
                results.last.max_log_likelihood_full_fits,
            )
        )

        hyper_result = copy.deepcopy(results.last)
        hyper_result.model = hyper_result.model.copy_with_fixed_priors(
            hyper_result.instance
        )

        hyper_result.analysis.uses_hyper_images = True
        hyper_result.analysis.model_images = model_images

        phase.search.model.parallel_trap = []
        phase.search.model.parallel_ccd = []
        phase.search.tag = ""

        phase.search.const_efficiency_mode = conf.instance.non_linear.get(
            "MultiNest", "const_efficiency_mode", bool
        )
        phase.search.facc = conf.instance.non_linear.get(
            "MultiNest", "sampling_efficiency", float
        )
        phase.search.n_live_points = conf.instance.non_linear.get(
            "MultiNest", "n_live_points", int
        )
        phase.search.multimodal = conf.instance.non_linear.get(
            "MultiNest", "multimodal", bool
        )

        analysis = self.Analysis(
            masked_ci_dataset_full=masked_ci_dataset_full, model_images=model_images
        )

        result = phase.search.f(analysis)

        def transfer_field(name):
            if hasattr(result.instance, name):
                setattr(hyper_result.instance, name, getattr(result.instance, name))
                setattr(hyper_result.model, name, getattr(result.model, name))

        transfer_field(name="hyper_noise_scalar_of_ci_regions")
        transfer_field(name="hyper_noise_scalar_of_parallel_trails")
        transfer_field(name="hyper_noise_scalar_of_serial_trails")
        transfer_field(name="hyper_noise_scalar_of_serial_overscan_no_trails")

        return hyper_result
