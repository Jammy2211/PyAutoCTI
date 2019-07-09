from functools import partial

import numpy as np
import autofit as af

from autocti import exc
from autocti.pipeline import tagging as tag
from autocti.charge_injection import ci_fit, ci_data, ci_hyper, ci_mask
from autocti.charge_injection.plotters import ci_data_plotters, ci_fit_plotters
from autocti.data import mask as msk
from autocti.model import arctic_params


def cti_params_for_instance(instance):
    return arctic_params.ArcticParams(
        parallel_ccd=instance.parallel_ccd if hasattr(instance, "parallel_ccd") else None,
        parallel_species=instance.parallel_species if hasattr(instance, "parallel_species") else None,
        serial_ccd=instance.serial_ccd if hasattr(instance, "serial_ccd") else None,
        serial_species=instance.serial_species if hasattr(instance, "serial_species") else None)


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class Phase(af.AbstractPhase):

    def make_result(self, result, analysis):
        return self.__class__.Result(constant=result.constant, figure_of_merit=result.figure_of_merit,
                                     previous_variable=result.previous_variable, gaussian_tuples=result.gaussian_tuples,
                                     analysis=analysis, optimizer=self.optimizer)

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), optimizer_class=af.DownhillSimplex,
                 mask_function=msk.Mask.empty_for_shape, columns=None, rows=None,
                 parallel_front_edge_mask_rows=None, parallel_trails_mask_rows=None,
                 serial_front_edge_mask_columns=None, serial_trails_mask_columns=None,
                 parallel_total_density_range=None, serial_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase in an analysis pipeline. Uses the set NonLinear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a NonLinear optimizer
            The side length of the subgrid
        """

        self.mask_function = mask_function
        self.columns = columns
        self.rows = rows
        self.parallel_front_edge_mask_rows = parallel_front_edge_mask_rows
        self.parallel_trails_mask_rows = parallel_trails_mask_rows
        self.serial_front_edge_mask_columns = serial_front_edge_mask_columns
        self.serial_trails_mask_columns = serial_trails_mask_columns
        self.parallel_total_density_range = parallel_total_density_range
        self.serial_total_density_range = serial_total_density_range
        self.cosmic_ray_parallel_buffer = cosmic_ray_parallel_buffer
        self.cosmic_ray_serial_buffer = cosmic_ray_serial_buffer
        self.cosmic_ray_diagonal_buffer = cosmic_ray_diagonal_buffer

        if tag_phases:

            phase_tag = tag.phase_tag_from_phase_settings(columns=columns, rows=rows,
                                                          parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
                                                          parallel_trails_mask_rows=parallel_trails_mask_rows,
                                                          serial_front_edge_mask_columns=serial_front_edge_mask_columns,
                                                          serial_trails_mask_columns=serial_trails_mask_columns,
                                                          parallel_total_density_range=self.parallel_total_density_range,
                                                          serial_total_density_range=self.serial_total_density_range,
                                                          cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                                                          cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                                                          cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)

        else:

            phase_tag = None

        super().__init__(phase_name=phase_name, phase_tag=phase_tag, phase_folders=phase_folders,
                         optimizer_class=optimizer_class)

    @property
    def variable(self):
        """
        Convenience method

        Returns
        -------
        ModelMapper
            A model mapper comprising all the variable (prior) objects in this analysis
        """
        return self.optimizer.variable

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
        result: Phase.Result
            A result object comprising the best fit model and other ci_data.
        """

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, results=results,
                                      pool=pool)
        self.pass_priors(results)
        self.assert_and_save_pickle()
        result = self.optimizer.fit(analysis)

        return self.make_result(result=result, analysis=analysis)

    # noinspection PyMethodMayBeStatic
    def extract_ci_data(self, data, mask):
        return ci_data.MaskedCIData(image=data.image, noise_map=data.noise_map, ci_pre_cti=data.ci_pre_cti, mask=mask,
                                    ci_pattern=data.ci_pattern, ci_frame=data.ci_frame)

    def masks_for_analysis_from_ci_datas(self, ci_datas):

        masks = list(map(lambda data: self.mask_function(shape=data.image.shape, ci_frame=data.ci_frame), ci_datas))

        cosmic_ray_masks = list(map(lambda data: msk.Mask.from_cosmic_ray_image(
            shape=data.shape, frame_geometry=data.ci_frame.frame_geometry,
            cosmic_ray_image=data.cosmic_ray_image,
            cosmic_ray_parallel_buffer=self.cosmic_ray_parallel_buffer,
            cosmic_ray_serial_buffer=self.cosmic_ray_serial_buffer,
            cosmic_ray_diagonal_buffer=self.cosmic_ray_diagonal_buffer)

        if data.cosmic_ray_image is not None else None, ci_datas))

        masks = list(map(lambda mask, cosmic_ray_mask:
                         mask + cosmic_ray_mask if cosmic_ray_mask is not None else mask,
                         masks, cosmic_ray_masks))

        if self.parallel_front_edge_mask_rows is not None:
            parallel_front_edge_masks = list(map(lambda data:
                                                 ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(
                                                     shape=data.shape, ci_frame=data.ci_frame,
                                                     rows=self.parallel_front_edge_mask_rows),
                                                 ci_datas))

            masks = list(map(lambda mask, parallel_front_edge_mask: mask + parallel_front_edge_mask,
                             masks, parallel_front_edge_masks))

        if self.parallel_trails_mask_rows is not None:
            parallel_trails_masks = list(map(lambda data:
                                             ci_mask.CIMask.masked_parallel_trails_from_ci_frame(shape=data.shape,
                                                                                                 ci_frame=data.ci_frame,
                                                                                                 rows=self.parallel_trails_mask_rows),
                                             ci_datas))

            masks = list(map(lambda mask, parallel_trails_mask: mask + parallel_trails_mask,
                             masks, parallel_trails_masks))

        if self.serial_front_edge_mask_columns is not None:
            serial_front_edge_masks = list(map(lambda data:
                                               ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(
                                                   shape=data.shape, ci_frame=data.ci_frame,
                                                   columns=self.serial_front_edge_mask_columns),
                                               ci_datas))

            masks = list(map(lambda mask, serial_front_edge_mask:
                             mask + serial_front_edge_mask,
                             masks, serial_front_edge_masks))

        if self.serial_trails_mask_columns is not None:
            serial_trails_masks = list(map(lambda data:
                                           ci_mask.CIMask.masked_serial_trails_from_ci_frame(
                                               shape=data.shape, ci_frame=data.ci_frame,
                                               columns=self.serial_trails_mask_columns),
                                           ci_datas))

            masks = list(map(lambda mask, serial_trails_mask: mask + serial_trails_mask,
                             masks, serial_trails_masks))

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

        ci_datas_fit = [self.extract_ci_data(data=data, mask=mask) for data, mask in zip(ci_datas, masks)]
        ci_datas_full = list(map(lambda data, mask:
                                 ci_data.MaskedCIData(image=data.image, noise_map=data.noise_map,
                                                      ci_pre_cti=data.ci_pre_cti, mask=mask,
                                                      ci_pattern=data.ci_pattern, ci_frame=data.ci_frame),
                                 ci_datas, masks))

        analysis = self.__class__.Analysis(ci_datas_extracted=ci_datas_fit, ci_datas_full=ci_datas_full,
                                           cti_settings=cti_settings,
                                           parallel_total_density_range=self.parallel_total_density_range,
                                           serial_total_density_range=self.serial_total_density_range,
                                           phase_name=self.phase_name, results=results, pool=pool)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(af.Analysis):

        def __init__(self, ci_datas_extracted, ci_datas_full, cti_settings,
                     serial_total_density_range, parallel_total_density_range, phase_name, results=None,
                     pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            self.ci_datas_extracted = ci_datas_full
            self.ci_datas_full = ci_datas_full
            self.cti_settings = cti_settings
            self.parallel_total_density_range = parallel_total_density_range
            self.serial_total_density_range = serial_total_density_range
            self.phase_name = phase_name
            self.phase_output_path = "{}/{}".format(af.conf.instance.output_path, self.phase_name)

            self.pool = pool or ConsecutivePool
            self.results = results

            self.plot_count = 0

            self.extract_array_from_mask = af.conf.instance.visualize.get('figures', 'extract_images_from_mask', bool)

            def output_plots(name):
                return af.conf.instance.visualize.get('plots', name, bool)

            self.plot_ci_data_as_subplot = output_plots('plot_ci_data_as_subplot')
            self.plot_ci_data_image = output_plots('plot_ci_data_image')
            self.plot_ci_data_noise_map = output_plots('plot_ci_data_noise_map')
            self.plot_ci_data_ci_pre_cti = output_plots('plot_ci_data_ci_pre_cti')
            self.plot_ci_data_signal_to_noise_map = output_plots('plot_ci_data_signal_to_noise_map')

            self.plot_ci_fit_all_at_end_png = output_plots('plot_ci_fit_all_at_end_png')
            self.plot_ci_fit_all_at_end_fits = output_plots('plot_ci_fit_all_at_end_fits')

            self.plot_ci_fit_as_subplot = output_plots('plot_ci_fit_as_subplot')
            self.plot_ci_fit_residual_maps_subplot = output_plots('plot_ci_fit_residual_maps_subplot')
            self.plot_ci_fit_chi_squared_maps_subplot = output_plots('plot_ci_fit_chi_squared_maps_subplot')
            
            self.plot_ci_fit_image = output_plots('plot_ci_fit_image')
            self.plot_ci_fit_noise_map = output_plots('plot_ci_fit_noise_map')
            self.plot_ci_fit_signal_to_noise_map = output_plots('plot_ci_fit_signal_to_noise_map')
            self.plot_ci_fit_ci_pre_cti = output_plots('plot_ci_fit_ci_pre_cti')
            self.plot_ci_fit_ci_post_cti = output_plots('plot_ci_fit_ci_post_cti')
            self.plot_ci_fit_residual_map = output_plots('plot_ci_fit_residual_map')
            self.plot_ci_fit_chi_squared_map = output_plots('plot_ci_fit_chi_squared_map')
            self.plot_ci_fit_noise_scaling_maps = output_plots('plot_ci_fit_noise_scaling_maps')

            self.plot_parallel_front_edge_line = output_plots('plot_parallel_front_edge_line')
            self.plot_parallel_trails_line = output_plots('plot_parallel_trails_line')
            self.plot_serial_front_edge_line = output_plots('plot_serial_front_edge_line')
            self.plot_serial_trails_line = output_plots('plot_serial_trails_line')

            self.is_hyper = False

        @property
        def last_results(self):
            if self.results is not None:
                return self.results.last

        def visualize(self, instance, image_path, during_analysis):

            ci_data_plotters.plot_ci_data_for_phase(
                ci_datas_extracted=self.ci_datas_extracted,
                extract_array_from_mask=self.extract_array_from_mask,
                should_plot_as_subplot=self.plot_ci_data_as_subplot,
                should_plot_image=self.plot_ci_data_image,
                should_plot_noise_map=self.plot_ci_data_noise_map,
                should_plot_ci_pre_cti=self.plot_ci_data_ci_pre_cti,
                should_plot_signal_to_noise_map=self.plot_ci_data_signal_to_noise_map,
                should_plot_parallel_front_edge_line=self.plot_parallel_front_edge_line,
                should_plot_parallel_trails_line=self.plot_parallel_trails_line,
                should_plot_serial_front_edge_line=self.plot_serial_front_edge_line,
                should_plot_serial_trails_line=self.plot_serial_trails_line,
                visualize_path=image_path)

            if not self.is_hyper:
                fits = self.fits_of_ci_data_extracted_for_instance(instance=instance)
            elif self.is_hyper:
                fits = self.fits_of_ci_data_hyper_extracted_for_instance(instance=instance)

            ci_fit_plotters.plot_ci_fit_for_phase(
                fits=fits, during_analysis=during_analysis,
                extract_array_from_mask=self.extract_array_from_mask,
                should_plot_all_at_end_png=self.plot_ci_fit_all_at_end_png,
                should_plot_all_at_end_fits=self.plot_ci_fit_all_at_end_fits,
                should_plot_as_subplot=self.plot_ci_fit_as_subplot,
                should_plot_residual_maps_subplot=self.plot_ci_fit_residual_maps_subplot,
                should_plot_chi_squared_maps_subplot=self.plot_ci_fit_chi_squared_maps_subplot,
                should_plot_image=self.plot_ci_fit_image,
                should_plot_noise_map=self.plot_ci_fit_noise_map,
                should_plot_signal_to_noise_map=self.plot_ci_fit_signal_to_noise_map,
                should_plot_ci_pre_cti=self.plot_ci_fit_ci_pre_cti,
                should_plot_ci_post_cti=self.plot_ci_fit_ci_post_cti,
                should_plot_residual_map=self.plot_ci_fit_residual_map,
                should_plot_chi_squared_map=self.plot_ci_fit_chi_squared_map,
                should_plot_noise_scaling_maps=self.plot_ci_fit_noise_scaling_maps,
                should_plot_parallel_front_edge_line=self.plot_parallel_front_edge_line,
                should_plot_parallel_trails_line=self.plot_parallel_trails_line,
                should_plot_serial_front_edge_line=self.plot_serial_front_edge_line,
                should_plot_serial_trails_line=self.plot_serial_trails_line,
                visualize_path=image_path)

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
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            likelihood = np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_extracted)))
            return likelihood

        def check_trap_lifetimes_are_ascending(self, cti_params):
            raise NotImplementedError

        def check_total_density_within_range(self, cti_params):
            raise NotImplementedError

        def fits_of_ci_data_extracted_for_instance(self, instance):

            cti_params = cti_params_for_instance(instance=instance)
            return list(map(lambda ci_data_fit:
                            ci_fit.CIFit(
                                masked_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=self.cti_settings),
                            self.ci_datas_extracted))

        def fits_of_ci_data_full_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance=instance)
            return list(map(lambda data:
                            ci_fit.CIFit(
                                masked_ci_data=data, cti_params=cti_params, cti_settings=self.cti_settings),
                            self.ci_datas_full))

        def fits_of_ci_data_hyper_extracted_for_instance(self, instance):
            raise NotImplementedError

        def fits_of_ci_data_hyper_full_for_instance(self, instance):
            raise NotImplementedError

    class Result(af.Result):

        # noinspection PyUnusedLocal
        def __init__(self, constant, figure_of_merit, previous_variable, gaussian_tuples, analysis, optimizer):
            """
            The result of a phase
            """

            super(Phase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit,
                                               previous_variable=previous_variable, gaussian_tuples=gaussian_tuples)

            self.analysis = analysis
            self.optimizer = optimizer

        @property
        def most_likely_extracted_fits(self):
            return self.analysis.fits_of_ci_data_extracted_for_instance(instance=self.constant)

        @property
        def most_likely_full_fits(self):
            return self.analysis.fits_of_ci_data_full_for_instance(instance=self.constant)

        @property
        def noise_scaling_maps_of_ci_regions(self):
            return list(map(lambda most_likely_full_fit:
                            most_likely_full_fit.noise_scaling_map_of_ci_regions,
                            self.most_likely_full_fits))

        @property
        def noise_scaling_maps_of_parallel_trails(self):
            return list(map(lambda most_likely_full_fit:
                            most_likely_full_fit.noise_scaling_map_of_parallel_trails,
                            self.most_likely_full_fits))

        @property
        def noise_scaling_maps_of_serial_trails(self):
            return list(map(lambda most_likely_full_fit:
                            most_likely_full_fit.noise_scaling_map_of_serial_trails,
                            self.most_likely_full_fits))

        @property
        def noise_scaling_maps_of_serial_overscan_above_trails(self):
            return list(map(lambda most_likely_full_fit:
                            most_likely_full_fit.noise_scaling_map_of_serial_overscan_above_trails,
                            self.most_likely_full_fits))


class ParallelPhase(Phase):

    parallel_species = af.PhaseProperty("parallel_species")
    parallel_ccd = af.PhaseProperty("parallel_ccd")

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), parallel_species=(), parallel_ccd=None,
                 optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape, columns=None,
                 parallel_front_edge_mask_rows=None, parallel_trails_mask_rows=None,
                 parallel_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model
        """

        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         optimizer_class=optimizer_class, mask_function=mask_function, columns=columns, rows=None,
                         parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
                         parallel_trails_mask_rows=parallel_trails_mask_rows,
                         parallel_total_density_range=parallel_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)

        self.parallel_species = parallel_species
        self.parallel_ccd = parallel_ccd

    def extract_ci_data(self, data, mask):
        return data.parallel_calibration_data(
            columns=(0, self.columns or data.ci_frame.frame_geometry.parallel_overscan.total_columns),
            mask=mask)

    class Analysis(Phase.Analysis):

        def __init__(self, ci_datas_extracted, ci_datas_full, cti_settings,
                     serial_total_density_range, parallel_total_density_range, phase_name, results=None,
                     pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            super().__init__(ci_datas_extracted=ci_datas_extracted, ci_datas_full=ci_datas_full,
                             cti_settings=cti_settings,
                             serial_total_density_range=serial_total_density_range,
                             parallel_total_density_range=parallel_total_density_range, phase_name=phase_name,
                             results=results, pool=pool)

            self.plot_serial_front_edge_line = False
            self.plot_serial_trails_line = False

        def check_trap_lifetimes_are_ascending(self, cti_params):

            trap_lifetimes = [parallel_species.trap_lifetime for parallel_species in cti_params.parallel_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

        def check_total_density_within_range(self, cti_params):

            if self.parallel_total_density_range is not None:

                total_density = sum([parallel_species.trap_density for parallel_species in cti_params.parallel_species])

                if total_density < self.parallel_total_density_range[0] or total_density > self.parallel_total_density_range[1]:
                    raise exc.PriorException

        def describe(self, instance):
            return ("\nRunning CTI analysis for... \n\n"
                    "Parallel CTI: \n"
                    "Parallel Species:\n{}\n\n"
                    "Parallel CCD\n{}\n\n".format(instance.parallel_species, instance.parallel_ccd))


class SerialPhase(Phase):
    
    serial_species = af.PhaseProperty("serial_species")
    serial_ccd = af.PhaseProperty("serial_ccd")

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), serial_species=(), serial_ccd=None,
                 optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape, rows=None,
                 serial_front_edge_mask_columns=None, serial_trails_mask_columns=None,
                 serial_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model
        """

        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         optimizer_class=optimizer_class, mask_function=mask_function, rows=rows,
                         serial_front_edge_mask_columns=serial_front_edge_mask_columns,
                         serial_trails_mask_columns=serial_trails_mask_columns,
                         serial_total_density_range=serial_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)

        self.serial_species = serial_species
        self.serial_ccd = serial_ccd

    def extract_ci_data(self, data, mask):
        return data.serial_calibration_data(rows=self.rows or (0, data.ci_pattern.regions[0].total_rows),
                                            mask=mask)

    class Analysis(Phase.Analysis):

        def __init__(self, ci_datas_extracted, ci_datas_full, cti_settings,
                     serial_total_density_range, parallel_total_density_range, phase_name, results=None,
                     pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            super().__init__(ci_datas_extracted=ci_datas_extracted, ci_datas_full=ci_datas_full,
                             cti_settings=cti_settings,
                             serial_total_density_range=serial_total_density_range,
                             parallel_total_density_range=parallel_total_density_range, phase_name=phase_name,
                             results=results, pool=pool)

            self.plot_parallel_front_edge_line = False
            self.plot_parallel_trails_line = False

        def check_trap_lifetimes_are_ascending(self, cti_params):
            trap_lifetimes = [serial_species.trap_lifetime for serial_species in cti_params.serial_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

        def check_total_density_within_range(self, cti_params):

            if self.serial_total_density_range is not None:

                total_density = sum([serial_species.trap_density for serial_species in cti_params.serial_species])
    
                if total_density < self.serial_total_density_range[0] or total_density > self.serial_total_density_range[1]:
                    raise exc.PriorException

        def describe(self, instance):
            return (
                "\nRunning CTI analysis for... \n\n"
                "Serial CTI: \n"
                "Serial Species:\n{}\n\n"
                "Serial CCD\n{}\n\n".format(instance.serial_species, instance.serial_ccd))


class ParallelSerialPhase(Phase):
    parallel_species = af.PhaseProperty("parallel_species")
    serial_species = af.PhaseProperty("serial_species")
    parallel_ccd = af.PhaseProperty("parallel_ccd")
    serial_ccd = af.PhaseProperty("serial_ccd")

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), parallel_species=(), serial_species=(),
                 parallel_ccd=None, serial_ccd=None,
                 optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape,
                 parallel_front_edge_mask_rows=None, parallel_trails_mask_rows=None,
                 serial_front_edge_mask_columns=None, serial_trails_mask_columns=None,
                 parallel_total_density_range=None, serial_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """

        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         optimizer_class=optimizer_class, mask_function=mask_function, columns=None, rows=None,
                         parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
                         parallel_trails_mask_rows=parallel_trails_mask_rows,
                         serial_front_edge_mask_columns=serial_front_edge_mask_columns,
                         serial_trails_mask_columns=serial_trails_mask_columns,
                         parallel_total_density_range=parallel_total_density_range,
                         serial_total_density_range=serial_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)

        self.parallel_species = parallel_species
        self.serial_species = serial_species
        self.parallel_ccd = parallel_ccd
        self.serial_ccd = serial_ccd

    def extract_ci_data(self, data, mask):
        return data.parallel_serial_calibration_data(mask)

    class Analysis(Phase.Analysis):

        def check_trap_lifetimes_are_ascending(self, cti_params):

            trap_lifetimes = [parallel_species.trap_lifetime for parallel_species in cti_params.parallel_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

            trap_lifetimes = [serial_species.trap_lifetime for serial_species in cti_params.serial_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

        def check_total_density_within_range(self, cti_params):

            if self.parallel_total_density_range is not None:

                total_density = sum([parallel_species.trap_density for parallel_species in cti_params.parallel_species])

                if total_density < self.parallel_total_density_range[0] or total_density > self.parallel_total_density_range[1]:
                    raise exc.PriorException

            if self.serial_total_density_range is not None:

                total_density = sum([serial_species.trap_density for serial_species in cti_params.serial_species])

                if total_density < self.serial_total_density_range[0] or total_density > self.serial_total_density_range[1]:
                    raise exc.PriorException

        def describe(self, instance):
            return (
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI: \n"
                "Parallel Species:\n{}\n\n"
                "Parallel CCD\n{}\n\n"
                "Serial CTI: \n"
                "Serial Species:\n{}\n\n"
                "Serial CCD\n{}\n\n".format(instance.parallel_species, instance.parallel_ccd,
                                            instance.serial_species, instance.serial_ccd))


class HyperPhase(Phase):
    """
    Mixin for hyper phases. Extracts noise scaling maps and creates MaskedCIHyperData objects for analysis.
    """
    hyper_noise_scalars = af.PhaseProperty("hyper_noise_scalars")

    def __init__(self, phase_name, phase_folders, *args, **kwargs):
        super().__init__(phase_name=phase_name, phase_folders=phase_folders, *args, **kwargs)

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        raise NotImplementedError()

    def noise_scaling_maps_from_result(self, result):
        raise NotImplementedError()

    def make_analysis(self, ci_datas, cti_settings, results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        cti_settings
        ci_datas
        pool
        results: [Result]
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """

        masks = self.masks_for_analysis_from_ci_datas(ci_datas=ci_datas)

        noise_scaling_maps = self.noise_scaling_maps_from_result(results[-1])

        ci_datas_fit = [self.extract_ci_hyper_data(data=data, mask=mask, noise_scaling_maps=maps) for
                        data, mask, maps in zip(ci_datas, masks, noise_scaling_maps)]
        ci_datas_full = list(map(lambda data, mask, maps:
                                 ci_data.MaskedCIHyperData(image=data.image, noise_map=data.noise_map,
                                                           ci_pre_cti=data.ci_pre_cti, mask=mask,
                                                           ci_pattern=data.ci_pattern, ci_frame=data.ci_frame,
                                                           noise_scaling_maps=maps),
                                 ci_datas, masks, noise_scaling_maps))

        class HyperAnalysis(self.__class__.Analysis):

            def __init__(self, ci_datas_extracted, ci_datas_full, cti_settings,
                         serial_total_density_range, parallel_total_density_range, phase_name, results=None,
                         pool=None):
                """
                An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
                set of objects describing a model and determines how well they fit the image.

                Params
                ----------
                ci_data : [CIImage.CIImage]
                    The charge injection ci_data-sets.
                """

                super().__init__(ci_datas_extracted=ci_datas_extracted, ci_datas_full=ci_datas_full,
                                 cti_settings=cti_settings,
                                 serial_total_density_range=serial_total_density_range,
                                 parallel_total_density_range=parallel_total_density_range, phase_name=phase_name,
                                 results=results, pool=pool)

                self.is_hyper = True

            def fits_of_ci_data_hyper_extracted_for_instance(self, instance):

                cti_params = cti_params_for_instance(instance=instance)
                return list(map(lambda ci_data_hyper_fit:
                                ci_fit.CIHyperFit(
                                    masked_hyper_ci_data=ci_data_hyper_fit, cti_params=cti_params,
                                    cti_settings=self.cti_settings,
                                    hyper_noise_scalars=instance.hyper_noise_scalars),
                                self.ci_datas_extracted))

            def fits_of_ci_data_hyper_full_for_instance(self, instance):
                cti_params = cti_params_for_instance(instance=instance)
                return list(map(lambda hyper_data:
                                ci_fit.CIHyperFit(
                                    masked_hyper_ci_data=hyper_data, cti_params=cti_params,
                                    cti_settings=self.cti_settings,
                                    hyper_noise_scalars=instance.hyper_noise_scalars),
                                self.ci_datas_full))

            def fit(self, instance):
                """
                Runs the analysis. Determine how well the supplied cti_params fits the image.

                Params
                ----------
                parallel : arctic_params.ParallelParams
                    A class describing the parallel cti model and parameters.
                parallel : arctic_params.SerialParams
                    A class describing the serial cti model and parameters.
                hyp : ci_hyper.HyperCINoise
                    A class describing the noises scaling ci_hyper-parameters.

                Returns
                -------
                result: Result
                    An object comprising the final cti_params instances generated and a corresponding figure_of_merit
                """
                cti_params = cti_params_for_instance(instance=instance)
                self.check_trap_lifetimes_are_ascending(cti_params=cti_params)
                pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                        hyper_noise_scalars=instance.hyper_noise_scalars)
                likelihood = np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_extracted)))
                return likelihood

            def describe(self, instance):
                return (
                    "{}"
                    "Hyper Parameters:\n{}\n\n".format(super().describe(instance),
                                                       " ".join(map(str, instance.hyper_noise_scalars))))

        analysis = HyperAnalysis(ci_datas_extracted=ci_datas_fit, ci_datas_full=ci_datas_full,
                                 cti_settings=cti_settings, phase_name=self.phase_name,
                                 parallel_total_density_range=self.parallel_total_density_range,
                                 serial_total_density_range=self.serial_total_density_range,
                                 results=results, pool=pool)
        return analysis

    @property
    def number_of_noise_scalars(self):
        return len(self.hyper_noise_scalars)

    @number_of_noise_scalars.setter
    def number_of_noise_scalars(self, number):
        self.hyper_noise_scalars = [af.PriorModel(ci_hyper.CIHyperNoiseScalar) for _ in range(number)]


class ParallelHyperPhase(ParallelPhase, HyperPhase):
    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), parallel_species=(), parallel_ccd=None,
                 optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape, columns=None,
                 parallel_front_edge_mask_rows=None, parallel_trails_mask_rows=None,
                 parallel_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """
        HyperPhase.__init__(self=self, phase_name=phase_name, phase_folders=phase_folders)
        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         parallel_species=parallel_species, parallel_ccd=parallel_ccd,
                         optimizer_class=optimizer_class, mask_function=mask_function, columns=columns,
                         parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
                         parallel_trails_mask_rows=parallel_trails_mask_rows,
                         parallel_total_density_range=parallel_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)
        self.number_of_noise_scalars = 2

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """

        noise_scaling_maps = []
        noise_scaling_maps_of_ci_regions = result.noise_scaling_maps_of_ci_regions
        noise_scaling_maps_of_parallel_trails = result.noise_scaling_maps_of_parallel_trails

        for image_index in range(len(noise_scaling_maps_of_ci_regions)):

            noise_scaling_maps.append([noise_scaling_maps_of_ci_regions[image_index],
                                      noise_scaling_maps_of_parallel_trails[image_index]])

        return noise_scaling_maps

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.parallel_hyper_calibration_data(
            columns=(0, self.columns or data.ci_frame.frame_geometry.parallel_overscan.total_columns),
            mask=mask, noise_scaling_maps=noise_scaling_maps)


class SerialHyperPhase(SerialPhase, HyperPhase):

    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), serial_species=(), serial_ccd=None,
                 optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape, rows=None,
                 serial_front_edge_mask_columns=None, serial_trails_mask_columns=None,
                 serial_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model
        """
        HyperPhase.__init__(self=self, phase_name=phase_name, phase_folders=phase_folders)
        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         serial_species=serial_species, serial_ccd=serial_ccd, optimizer_class=optimizer_class,
                         mask_function=mask_function, rows=rows,
                         serial_front_edge_mask_columns=serial_front_edge_mask_columns,
                         serial_trails_mask_columns=serial_trails_mask_columns,
                         serial_total_density_range=serial_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)
        self.number_of_noise_scalars = 2

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """
        noise_scaling_maps = []
        noise_scaling_maps_of_ci_regions = result.noise_scaling_maps_of_ci_regions
        noise_scaling_maps_of_serial_trails = result.noise_scaling_maps_of_serial_trails

        for image_index in range(len(noise_scaling_maps_of_ci_regions)):

            noise_scaling_maps.append([noise_scaling_maps_of_ci_regions[image_index],
                                      noise_scaling_maps_of_serial_trails[image_index]])

        return noise_scaling_maps

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.serial_hyper_calibration_data(rows=self.rows or (0, data.ci_pattern.regions[0].total_rows),
                                                  mask=mask, noise_scaling_maps=noise_scaling_maps)


class ParallelSerialHyperPhase(ParallelSerialPhase, HyperPhase):
    def __init__(self, phase_name, tag_phases=True, phase_folders=tuple(), parallel_species=(), serial_species=(), parallel_ccd=None,
                 serial_ccd=None, optimizer_class=af.MultiNest, mask_function=msk.Mask.empty_for_shape,
                 parallel_front_edge_mask_rows=None, parallel_trails_mask_rows=None,
                 serial_front_edge_mask_columns=None, serial_trails_mask_columns=None,
                 parallel_total_density_range=None, serial_total_density_range=None,
                 cosmic_ray_parallel_buffer=10, cosmic_ray_serial_buffer=10, cosmic_ray_diagonal_buffer=3):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """
        HyperPhase.__init__(self=self, phase_name=phase_name, phase_folders=phase_folders)
        super().__init__(phase_name=phase_name, tag_phases=tag_phases, phase_folders=phase_folders,
                         parallel_species=parallel_species, serial_species=serial_species, parallel_ccd=parallel_ccd,
                         serial_ccd=serial_ccd, optimizer_class=optimizer_class, mask_function=mask_function,
                         parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
                         parallel_trails_mask_rows=parallel_trails_mask_rows,
                         serial_front_edge_mask_columns=serial_front_edge_mask_columns,
                         serial_trails_mask_columns=serial_trails_mask_columns,
                         parallel_total_density_range=parallel_total_density_range,
                         serial_total_density_range=serial_total_density_range,
                         cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
                         cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
                         cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer)
        self.number_of_noise_scalars = 4

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """

        noise_scaling_maps = []
        noise_scaling_maps_of_ci_regions = result.noise_scaling_maps_of_ci_regions
        noise_scaling_maps_of_parallel_trails = result.noise_scaling_maps_of_parallel_trails
        noise_scaling_maps_of_serial_trails = result.noise_scaling_maps_of_serial_trails
        noise_scaling_maps_of_serial_overscan_above_trails = result.noise_scaling_maps_of_serial_overscan_above_trails

        for image_index in range(len(noise_scaling_maps_of_ci_regions)):

            noise_scaling_maps.append([noise_scaling_maps_of_ci_regions[image_index],
                                       noise_scaling_maps_of_parallel_trails[image_index],
                                       noise_scaling_maps_of_serial_trails[image_index],
                                       noise_scaling_maps_of_serial_overscan_above_trails[image_index]])

        return noise_scaling_maps

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.parallel_serial_hyper_calibration_data(mask, noise_scaling_maps=noise_scaling_maps)


def pipe_cti(ci_data_fit, cti_params, cti_settings):
    fit = ci_fit.CIFit(masked_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)
    return fit.figure_of_merit


def pipe_cti_hyper(ci_data_fit, cti_params, cti_settings, hyper_noise_scalars):
    fit = ci_fit.CIHyperFit(masked_hyper_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings,
                            hyper_noise_scalars=hyper_noise_scalars)
    return fit.figure_of_merit
