from functools import partial

import numpy as np
import autofit as af

from autocti import exc
from autocti.pipeline import phase_tagging as tag
from autocti.pipeline.phase.phase import Phase
from autocti.pipeline.phase import phase_extensions
from autocti.charge_injection import ci_imaging, ci_fit, ci_mask
from autocti.plot import ci_fit_plots, phase_plots
from autocti.structures import mask as msk
from arctic import model


def cti_params_for_instance(instance):
    return model.ArcticParams(
        parallel_ccd_volume=instance.parallel_ccd_volume
        if hasattr(instance, "parallel_ccd_volume")
        else None,
        parallel_traps=instance.parallel_traps
        if hasattr(instance, "parallel_traps")
        else None,
        serial_ccd_volume=instance.serial_ccd_volume
        if hasattr(instance, "serial_ccd_volume")
        else None,
        serial_traps=instance.serial_traps
        if hasattr(instance, "serial_traps")
        else None,
    )


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class PhaseCI(Phase):

    parallel_traps = af.PhaseProperty("parallel_traps")
    serial_traps = af.PhaseProperty("serial_traps")
    parallel_ccd_volume = af.PhaseProperty("parallel_ccd_volume")
    serial_ccd_volume = af.PhaseProperty("serial_ccd_volume")
    hyper_noise_scalar_of_ci_regions = af.PhaseProperty(
        "hyper_noise_scalar_of_ci_regions"
    )
    hyper_noise_scalar_of_parallel_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_parallel_trails"
    )
    hyper_noise_scalar_of_serial_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_serial_trails"
    )
    hyper_noise_scalar_of_serial_overscan_above_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_serial_overscan_above_trails"
    )

    def make_result(self, result, analysis):
        return self.__class__.Result(
            constant=result.constant,
            figure_of_merit=result.figure_of_merit,
            previous_variable=result.previous_variable,
            gaussian_tuples=result.gaussian_tuples,
            analysis=analysis,
            optimizer=self.optimizer,
        )

    def __init__(
        self,
        phase_name,
        phase_folders=tuple(),
        parallel_traps=(),
        parallel_ccd_volume=None,
        serial_traps=(),
        serial_ccd_volume=None,
        hyper_noise_scalar_of_ci_regions=None,
        hyper_noise_scalar_of_parallel_trails=None,
        hyper_noise_scalar_of_serial_trails=None,
        hyper_noise_scalar_of_serial_overscan_above_trails=None,
        optimizer_class=af.DownhillSimplex,
        mask_function=msk.Mask.unmasked,
        columns=None,
        rows=None,
        parallel_front_edge_mask_rows=None,
        parallel_trails_mask_rows=None,
        parallel_total_density_range=None,
        serial_front_edge_mask_columns=None,
        serial_trails_mask_columns=None,
        serial_total_density_range=None,
        cosmic_ray_parallel_buffer=10,
        cosmic_ray_serial_buffer=10,
        cosmic_ray_diagonal_buffer=3,
    ):
        """
        A phase in an analysis pipeline. Uses the set NonLinear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a NonLinear optimizer
            The side length of the subgrid
        """

        self.phase_folders = phase_folders

        self.mask_function = mask_function
        self.columns = columns
        self.rows = rows
        self.parallel_front_edge_mask_rows = parallel_front_edge_mask_rows
        self.parallel_trails_mask_rows = parallel_trails_mask_rows
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_front_edge_mask_columns = serial_front_edge_mask_columns
        self.serial_trails_mask_columns = serial_trails_mask_columns
        self.serial_total_density_range = serial_total_density_range
        self.cosmic_ray_parallel_buffer = cosmic_ray_parallel_buffer
        self.cosmic_ray_serial_buffer = cosmic_ray_serial_buffer
        self.cosmic_ray_diagonal_buffer = cosmic_ray_diagonal_buffer

        phase_tag = tag.phase_tag_from_phase_settings(
            columns=columns,
            rows=rows,
            parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
            parallel_trails_mask_rows=parallel_trails_mask_rows,
            serial_front_edge_mask_columns=serial_front_edge_mask_columns,
            serial_trails_mask_columns=serial_trails_mask_columns,
            parallel_total_density_range=self.parallel_total_density_range,
            serial_total_density_range=self.serial_total_density_range,
            cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
            cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
            cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer,
        )

        super().__init__(
            phase_name=phase_name,
            phase_tag=phase_tag,
            phase_folders=phase_folders,
            optimizer_class=optimizer_class,
        )

        self.parallel_traps = parallel_traps
        self.parallel_ccd_volume = parallel_ccd_volume
        self.serial_traps = serial_traps
        self.serial_ccd_volume = serial_ccd_volume

        self.hyper_noise_scalar_of_ci_regions = hyper_noise_scalar_of_ci_regions
        self.hyper_noise_scalar_of_parallel_trails = (
            hyper_noise_scalar_of_parallel_trails
        )
        self.hyper_noise_scalar_of_serial_trails = hyper_noise_scalar_of_serial_trails
        self.hyper_noise_scalar_of_serial_overscan_above_trails = (
            hyper_noise_scalar_of_serial_overscan_above_trails
        )

    def run(self, ci_datas, cti_settings, results=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        pool
        cti_settings
        ci_datas
        results: [Result]
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: PhaseCI.Result
            A result object comprising the best fit model and other ci_data.
        """

        analysis = self.make_analysis(
            ci_datas=ci_datas, cti_settings=cti_settings, results=results, pool=pool
        )
        self.variable = self.variable.populate(results)
        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

    def masks_for_analysis_from_ci_datas(self, ci_datas):

        masks = list(
            map(
                lambda data: self.mask_function(
                    shape=data.profile_image.shape, ci_frame=data.ci_frame
                ),
                ci_datas,
            )
        )

        cosmic_ray_masks = list(
            map(
                lambda data: msk.Mask.from_cosmic_ray_map(
                    shape_2d=data.shape,
                    frame_geometry=data.ci_frame.frame_geometry,
                    cosmic_ray_map=data.cosmic_ray_map,
                    cosmic_ray_parallel_buffer=self.cosmic_ray_parallel_buffer,
                    cosmic_ray_serial_buffer=self.cosmic_ray_serial_buffer,
                    cosmic_ray_diagonal_buffer=self.cosmic_ray_diagonal_buffer,
                )
                if data.cosmic_ray_map is not None
                else None,
                ci_datas,
            )
        )

        masks = list(
            map(
                lambda mask, cosmic_ray_mask: mask + cosmic_ray_mask
                if cosmic_ray_mask is not None
                else mask,
                masks,
                cosmic_ray_masks,
            )
        )

        if self.parallel_front_edge_mask_rows is not None:
            parallel_front_edge_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        rows=self.parallel_front_edge_mask_rows,
                    ),
                    ci_datas,
                )
            )

            masks = list(
                map(
                    lambda mask, parallel_front_edge_mask: mask
                    + parallel_front_edge_mask,
                    masks,
                    parallel_front_edge_masks,
                )
            )

        if self.parallel_trails_mask_rows is not None:
            parallel_trails_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_parallel_trails_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        rows=self.parallel_trails_mask_rows,
                    ),
                    ci_datas,
                )
            )

            masks = list(
                map(
                    lambda mask, parallel_trails_mask: mask + parallel_trails_mask,
                    masks,
                    parallel_trails_masks,
                )
            )

        if self.serial_front_edge_mask_columns is not None:
            serial_front_edge_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        columns=self.serial_front_edge_mask_columns,
                    ),
                    ci_datas,
                )
            )

            masks = list(
                map(
                    lambda mask, serial_front_edge_mask: mask + serial_front_edge_mask,
                    masks,
                    serial_front_edge_masks,
                )
            )

        if self.serial_trails_mask_columns is not None:
            serial_trails_masks = list(
                map(
                    lambda data: ci_mask.CIMask.masked_serial_trails_from_ci_frame(
                        shape=data.shape,
                        ci_frame=data.ci_frame,
                        columns=self.serial_trails_mask_columns,
                    ),
                    ci_datas,
                )
            )

            masks = list(
                map(
                    lambda mask, serial_trails_mask: mask + serial_trails_mask,
                    masks,
                    serial_trails_masks,
                )
            )

        return masks

    def make_analysis(self, ci_datas, cti_settings, results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        cti_settings
        ci_datas
        pool
        results: [Results]
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """

        masks = self.masks_for_analysis_from_ci_datas(ci_datas=ci_datas)

        noise_scaling_maps = self.noise_scaling_maps_from_total_images_and_results(
            total_images=len(ci_datas), results=results
        )

        ci_datas_masked_extracted = [
            self.ci_datas_masked_extracted_from_ci_data(
                ci_data=data, mask=mask, noise_scaling_maps=maps
            )
            for data, mask, maps in zip(ci_datas, masks, noise_scaling_maps)
        ]

        ci_datas_masked_full = list(
            map(
                lambda data, mask, maps: ci_imaging.MaskedCIImaging(
                    image=data.profile_image,
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

        analysis = self.__class__.Analysis(
            ci_datas_masked_extracted=ci_datas_masked_extracted,
            ci_datas_masked_full=ci_datas_masked_full,
            cti_settings=cti_settings,
            parallel_total_density_range=self.parallel_total_density_range,
            serial_total_density_range=self.serial_total_density_range,
            phase_name=self.phase_name,
            results=results,
            pool=pool,
        )
        return analysis

    def noise_scaling_maps_from_total_images_and_results(self, total_images, results):

        if self.hyper_noise_scalar_of_ci_regions is not None:
            noise_scaling_maps_of_ci_regions = (
                results.last.noise_scaling_maps_of_ci_regions
            )
        else:
            noise_scaling_maps_of_ci_regions = total_images * [None]

        if self.hyper_noise_scalar_of_parallel_trails is not None:
            noise_scaling_maps_of_parallel_trails = (
                results.last.noise_scaling_maps_of_parallel_trails
            )
        else:
            noise_scaling_maps_of_parallel_trails = total_images * [None]

        if self.hyper_noise_scalar_of_serial_trails is not None:
            noise_scaling_maps_of_serial_trails = (
                results.last.noise_scaling_maps_of_serial_trails
            )
        else:
            noise_scaling_maps_of_serial_trails = total_images * [None]

        if self.hyper_noise_scalar_of_serial_overscan_above_trails is not None:
            noise_scaling_maps_of_serial_overscan_above_trails = (
                results.last.noise_scaling_maps_of_serial_overscan_above_trails
            )
        else:
            noise_scaling_maps_of_serial_overscan_above_trails = total_images * [None]

        noise_scaling_maps = []

        for image_index in range(total_images):
            noise_scaling_maps.append(
                [
                    noise_scaling_maps_of_ci_regions[image_index],
                    noise_scaling_maps_of_parallel_trails[image_index],
                    noise_scaling_maps_of_serial_trails[image_index],
                    noise_scaling_maps_of_serial_overscan_above_trails[image_index],
                ]
            )

        for image_index in range(total_images):
            noise_scaling_maps[image_index] = [
                noise_scaling_map
                for noise_scaling_map in noise_scaling_maps[image_index]
                if noise_scaling_map is not None
            ]

        noise_scaling_maps = list(filter(None, noise_scaling_maps))

        if noise_scaling_maps == []:
            noise_scaling_maps = total_images * [None]

        return noise_scaling_maps

    def customize_priors(self, results):
        """
        Perform any prior or constant passing. This could involve setting model
        attributes equal to priors or constants from a previous phase.

        Parameters
        ----------
        results: autofit.tools.pipeline.ResultsCollection
            The result of the previous phase
        """
        pass

    @property
    def is_only_parallel_fit(self):
        if self.parallel_ccd_volume is not None and self.serial_ccd_volume is None:
            return True
        else:
            return False

    @property
    def is_only_serial_fit(self):
        if self.parallel_ccd_volume is None and self.serial_ccd_volume is not None:
            return True
        else:
            return False

    @property
    def is_parallel_and_serial_fit(self):
        if self.parallel_ccd_volume is not None and self.serial_ccd_volume is not None:
            return True
        else:
            return False

    def ci_datas_masked_extracted_from_ci_data(
        self, ci_data, mask, noise_scaling_maps=None
    ):

        if self.is_only_parallel_fit:
            return ci_data.parallel_ci_data_masked_from_columns_and_mask(
                columns=(
                    0,
                    self.columns
                    or ci_data.ci_frame.frame_geometry.parallel_overscan.total_columns,
                ),
                mask=mask,
                noise_scaling_maps=noise_scaling_maps,
            )

        elif self.is_only_serial_fit:
            return ci_data.serial_ci_data_masked_from_rows_and_mask(
                rows=self.rows or (0, ci_data.ci_pattern.regions[0].total_rows),
                mask=mask,
                noise_scaling_maps=noise_scaling_maps,
            )
        elif self.is_parallel_and_serial_fit:
            return ci_data.parallel_serial_ci_data_masked_from_mask(
                mask=mask, noise_scaling_maps=noise_scaling_maps
            )

    @property
    def number_of_noise_scalars(self):
        return len(self.hyper_noise_scalars)

    @property
    def hyper_noise_scalars(self):
        return list(
            filter(
                None,
                [
                    self.hyper_noise_scalar_of_ci_regions,
                    self.hyper_noise_scalar_of_parallel_trails,
                    self.hyper_noise_scalar_of_serial_trails,
                    self.hyper_noise_scalar_of_serial_overscan_above_trails,
                ],
            )
        )

    def extend_with_hyper_noise_phases(self,):
        return phase_extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=(phase_extensions.HyperNoisePhase,)
        )

    # noinspection PyAbstractClass
    class Analysis(Phase.Analysis):
        def __init__(
            self,
            ci_datas_masked_extracted,
            ci_datas_masked_full,
            cti_settings,
            serial_total_density_range,
            parallel_total_density_range,
            phase_name,
            results=None,
            pool=None,
        ):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            self.ci_datas_masked_extracted = ci_datas_masked_extracted
            self.ci_datas_masked_full = ci_datas_masked_full
            self.cti_settings = cti_settings
            self.parallel_total_density_range = parallel_total_density_range
            self.serial_total_density_range = serial_total_density_range
            self.phase_name = phase_name
            self.phase_output_path = "{}/{}".format(
                af.conf.instance.output_path, self.phase_name
            )

            self.pool = pool or ConsecutivePool
            self.results = results

            self.plot_count = 0

            self.extract_array_from_mask = af.conf.instance.visualize.get(
                "figures", "extract_images_from_mask", bool
            )

            def output_plots(name):
                return af.conf.instance.visualize.get("plots", name, bool)

            self.plot_ci_data_as_subplot = output_plots("plot_ci_data_as_subplot")
            self.plot_ci_data_image = output_plots("plot_ci_data_image")
            self.plot_ci_data_noise_map = output_plots("plot_ci_data_noise_map")
            self.plot_ci_data_ci_pre_cti = output_plots("plot_ci_data_ci_pre_cti")
            self.plot_ci_data_signal_to_noise_map = output_plots(
                "plot_ci_data_signal_to_noise_map"
            )

            self.plot_ci_fit_all_at_end_png = output_plots("plot_ci_fit_all_at_end_png")
            self.plot_ci_fit_all_at_end_fits = output_plots(
                "plot_ci_fit_all_at_end_fits"
            )

            self.plot_ci_fit_as_subplot = output_plots("plot_ci_fit_as_subplot")
            self.plot_ci_fit_residual_maps_subplot = output_plots(
                "plot_ci_fit_residual_maps_subplot"
            )
            self.plot_ci_fit_chi_squared_maps_subplot = output_plots(
                "plot_ci_fit_chi_squared_maps_subplot"
            )

            self.plot_ci_fit_image = output_plots("plot_ci_fit_image")
            self.plot_ci_fit_noise_map = output_plots("plot_ci_fit_noise_map")
            self.plot_ci_fit_signal_to_noise_map = output_plots(
                "plot_ci_fit_signal_to_noise_map"
            )
            self.plot_ci_fit_ci_pre_cti = output_plots("plot_ci_fit_ci_pre_cti")
            self.plot_ci_fit_ci_post_cti = output_plots("plot_ci_fit_ci_post_cti")
            self.plot_ci_fit_residual_map = output_plots("plot_ci_fit_residual_map")
            self.plot_ci_fit_chi_squared_map = output_plots(
                "plot_ci_fit_chi_squared_map"
            )
            self.plot_ci_fit_noise_scaling_maps = output_plots(
                "plot_ci_fit_noise_scaling_maps"
            )

            self.plot_parallel_front_edge_line = output_plots(
                "plot_parallel_front_edge_line"
            )
            self.plot_parallel_trails_line = output_plots("plot_parallel_trails_line")
            self.plot_serial_front_edge_line = output_plots(
                "plot_serial_front_edge_line"
            )
            self.plot_serial_trails_line = output_plots("plot_serial_trails_line")

            self.is_hyper = False

        @property
        def last_results(self):
            if self.results is not None:
                return self.results.last

        def visualize(self, instance, image_path, during_analysis):

            phase_plots.plot_ci_data_for_phase(
                ci_datas_extracted=self.ci_datas_masked_extracted,
                extract_array_from_mask=self.extract_array_from_mask,
                plot_as_subplot=self.plot_ci_data_as_subplot,
                plot_image=self.plot_ci_data_image,
                plot_noise_map=self.plot_ci_data_noise_map,
                plot_ci_pre_cti=self.plot_ci_data_ci_pre_cti,
                plot_signal_to_noise_map=self.plot_ci_data_signal_to_noise_map,
                plot_parallel_front_edge_line=self.plot_parallel_front_edge_line,
                plot_parallel_trails_line=self.plot_parallel_trails_line,
                plot_serial_front_edge_line=self.plot_serial_front_edge_line,
                plot_serial_trails_line=self.plot_serial_trails_line,
                visualize_path=image_path,
            )

            if not self.is_hyper:
                fits = self.fits_of_ci_data_extracted_for_instance(instance=instance)
            elif self.is_hyper:
                fits = self.fits_of_ci_data_hyper_extracted_for_instance(
                    instance=instance
                )

            ci_fit_plots.plot_ci_fit_for_phase(
                fits=fits,
                during_analysis=during_analysis,
                extract_array_from_mask=self.extract_array_from_mask,
                plot_all_at_end_png=self.plot_ci_fit_all_at_end_png,
                plot_all_at_end_fits=self.plot_ci_fit_all_at_end_fits,
                plot_as_subplot=self.plot_ci_fit_as_subplot,
                plot_residual_maps_subplot=self.plot_ci_fit_residual_maps_subplot,
                plot_chi_squared_maps_subplot=self.plot_ci_fit_chi_squared_maps_subplot,
                plot_image=self.plot_ci_fit_image,
                plot_noise_map=self.plot_ci_fit_noise_map,
                plot_signal_to_noise_map=self.plot_ci_fit_signal_to_noise_map,
                plot_ci_pre_cti=self.plot_ci_fit_ci_pre_cti,
                plot_ci_post_cti=self.plot_ci_fit_ci_post_cti,
                plot_residual_map=self.plot_ci_fit_residual_map,
                plot_chi_squared_map=self.plot_ci_fit_chi_squared_map,
                plot_noise_scaling_maps=self.plot_ci_fit_noise_scaling_maps,
                plot_parallel_front_edge_line=self.plot_parallel_front_edge_line,
                plot_parallel_trails_line=self.plot_parallel_trails_line,
                plot_serial_front_edge_line=self.plot_serial_front_edge_line,
                plot_serial_trails_line=self.plot_serial_trails_line,
                visualize_path=image_path,
            )

        def fit(self, instance):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            instance

            Returns
            -------
            fit: ci_fit.Fit
                How fit the model is and the model
            """
            cti_params = cti_params_for_instance(instance=instance)
            self.check_trap_lifetimes_are_ascending(cti_params=cti_params)
            self.check_total_density_within_range(cti_params=cti_params)

            hyper_noise_scalars = self.hyper_noise_scalars_from_instance(
                instance=instance
            )

            pipe_cti_pass = partial(
                pipe_cti,
                cti_params=cti_params,
                cti_settings=self.cti_settings,
                hyper_noise_scalars=hyper_noise_scalars,
            )
            likelihood = np.sum(
                list(self.pool.map(pipe_cti_pass, self.ci_datas_masked_extracted))
            )
            return likelihood

        def check_trap_lifetimes_are_ascending(self, cti_params):

            if cti_params.parallel_traps:

                trap_lifetimes = [
                    parallel_traps.trap_lifetime
                    for parallel_traps in cti_params.parallel_traps
                ]

                if not sorted(trap_lifetimes) == trap_lifetimes:
                    raise exc.PriorException

            if cti_params.serial_traps:

                trap_lifetimes = [
                    serial_traps.trap_lifetime
                    for serial_traps in cti_params.serial_traps
                ]

                if not sorted(trap_lifetimes) == trap_lifetimes:
                    raise exc.PriorException

        def check_total_density_within_range(self, cti_params):

            if self.parallel_total_density_range is not None:

                total_density = sum(
                    [
                        parallel_traps.trap_density
                        for parallel_traps in cti_params.parallel_traps
                    ]
                )

                if (
                    total_density < self.parallel_total_density_range[0]
                    or total_density > self.parallel_total_density_range[1]
                ):
                    raise exc.PriorException

            if self.serial_total_density_range is not None:

                total_density = sum(
                    [
                        serial_traps.trap_density
                        for serial_traps in cti_params.serial_traps
                    ]
                )

                if (
                    total_density < self.serial_total_density_range[0]
                    or total_density > self.serial_total_density_range[1]
                ):
                    raise exc.PriorException

        def hyper_noise_scalars_from_instance(self, instance):
            hyper_noise_scalars = list(
                filter(
                    None,
                    [
                        instance.hyper_noise_scalar_of_ci_regions,
                        instance.hyper_noise_scalar_of_parallel_trails,
                        instance.hyper_noise_scalar_of_serial_trails,
                        instance.hyper_noise_scalar_of_serial_overscan_above_trails,
                    ],
                )
            )
            if hyper_noise_scalars:
                return hyper_noise_scalars
            else:
                return None

        def hyper_noise_scalars_from_instance_and_hyper_noise_scale(
            self, instance, hyper_noise_scale
        ):

            if hyper_noise_scale:
                return self.hyper_noise_scalars_from_instance(instance=instance)
            else:
                return None

        def fits_of_ci_data_extracted_for_instance(
            self, instance, hyper_noise_scale=True
        ):

            cti_params = cti_params_for_instance(instance=instance)
            hyper_noise_scalars = self.hyper_noise_scalars_from_instance_and_hyper_noise_scale(
                instance=instance, hyper_noise_scale=hyper_noise_scale
            )

            return list(
                map(
                    lambda ci_data_masked_extracted: ci_fit.CIImagingFit(
                        masked_ci_imaging=ci_data_masked_extracted,
                        cti_params=cti_params,
                        cti_settings=self.cti_settings,
                        hyper_noise_scalars=hyper_noise_scalars,
                    ),
                    self.ci_datas_masked_extracted,
                )
            )

        def fits_of_ci_data_full_for_instance(self, instance, hyper_noise_scale=True):

            cti_params = cti_params_for_instance(instance=instance)
            hyper_noise_scalars = self.hyper_noise_scalars_from_instance_and_hyper_noise_scale(
                instance=instance, hyper_noise_scale=hyper_noise_scale
            )

            return list(
                map(
                    lambda ci_data_masked_full: ci_fit.CIImagingFit(
                        masked_ci_imaging=ci_data_masked_full,
                        cti_params=cti_params,
                        cti_settings=self.cti_settings,
                        hyper_noise_scalars=hyper_noise_scalars,
                    ),
                    self.ci_datas_masked_full,
                )
            )

        def describe(self, instance):
            return (
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI: \n"
                "Parallel Trap:\n{}\n\n"
                "Parallel CCD\n{}\n\n"
                "Serial CTI: \n"
                "Serial Trap:\n{}\n\n"
                "Serial CCD\n{}\n\n".format(
                    instance.parallel_traps,
                    instance.parallel_ccd_volume,
                    instance.serial_traps,
                    instance.serial_ccd_volume,
                )
            )

    class Result(Phase.Result):

        # noinspection PyUnusedLocal
        def __init__(
            self,
            constant,
            figure_of_merit,
            previous_variable,
            gaussian_tuples,
            analysis,
            optimizer,
        ):
            """
            The result of a phase
            """

            super(PhaseCI.Result, self).__init__(
                constant=constant,
                figure_of_merit=figure_of_merit,
                previous_variable=previous_variable,
                gaussian_tuples=gaussian_tuples,
                analysis=analysis,
                optimizer=optimizer,
            )

            self.analysis = analysis
            self.optimizer = optimizer

        @property
        def most_likely_extracted_fits(self):
            return self.analysis.fits_of_ci_data_extracted_for_instance(
                instance=self.constant
            )

        @property
        def most_likely_full_fits(self):
            return self.analysis.fits_of_ci_data_full_for_instance(
                instance=self.constant
            )

        @property
        def most_likely_full_fits_no_hyper_scaling(self):
            return self.analysis.fits_of_ci_data_full_for_instance(
                instance=self.constant
            )

        @property
        def noise_scaling_maps_of_ci_regions(self):

            return list(
                map(
                    lambda fit: fit.chi_squared_map_of_ci_regions,
                    self.most_likely_full_fits_no_hyper_scaling,
                )
            )

        @property
        def noise_scaling_maps_of_parallel_trails(self):

            return list(
                map(
                    lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_parallel_trails,
                    self.most_likely_full_fits_no_hyper_scaling,
                )
            )

        @property
        def noise_scaling_maps_of_serial_trails(self):

            return list(
                map(
                    lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_serial_trails,
                    self.most_likely_full_fits_no_hyper_scaling,
                )
            )

        @property
        def noise_scaling_maps_of_serial_overscan_above_trails(self):

            return list(
                map(
                    lambda most_likely_full_fit: most_likely_full_fit.chi_squared_map_of_serial_overscan_above_trails,
                    self.most_likely_full_fits_no_hyper_scaling,
                )
            )


def pipe_cti(ci_data_masked, cti_params, cti_settings, hyper_noise_scalars):
    fit = ci_fit.CIImagingFit(
        masked_ci_imaging=ci_data_masked,
        cti_params=cti_params,
        cti_settings=cti_settings,
        hyper_noise_scalars=hyper_noise_scalars,
    )
    return fit.figure_of_merit
