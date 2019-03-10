from functools import partial

import numpy as np
from autofit import conf
from autofit.mapper import prior_model as pm
from autofit.optimize import non_linear as nl
from autofit.tools import phase as ph
from autofit.tools import phase_property

from autocti import exc
from autocti.charge_injection import ci_fit, ci_data, ci_hyper
from autocti.charge_injection.plotters import ci_data_plotters
from autocti.charge_injection.plotters import ci_fit_plotters
from autocti.data import mask as msk
from autocti.model import arctic_params


def cti_params_for_instance(instance):
    return arctic_params.ArcticParams(
        parallel_ccd=instance.parallel_ccd if hasattr(instance, "parallel_ccd") else None,
        parallel_species=instance.parallel_species if hasattr(instance, "parallel_species") else None,
        serial_ccd=instance.serial_ccd if hasattr(instance, "serial_ccd") else None,
        serial_species=instance.serial_species if hasattr(instance, "serial_species") else None
    )


class ConsecutivePool(object):
    """
    Replicates the interface of a multithread pool but performs computations consecutively
    """

    @staticmethod
    def map(func, ls):
        return map(func, ls)


class Phase(ph.AbstractPhase):

    def make_result(self, result, analysis):
        return self.__class__.Result(constant=result.constant, figure_of_merit=result.figure_of_merit,
                                     variable=result.variable, analysis=analysis, optimizer=self.optimizer)

    def __init__(self, optimizer_class=nl.DownhillSimplex, mask_function=msk.Mask.empty_for_shape, columns=None,
                 rows=None, phase_name=None):
        """
        A phase in an analysis pipeline. Uses the set NonLinear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a NonLinear optimizer
            The side length of the subgrid
        """
        super().__init__(optimizer_class, phase_name)
        self.mask_function = mask_function
        self.columns = columns
        self.rows = rows

    @property
    def constant(self):
        """
        Convenience method

        Returns
        -------
        ModelInstance
            A model instance comprising all the constant objects in this analysis
        """
        return self.optimizer.constant

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

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        pool
        cti_settings
        ci_datas
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed

        Returns
        -------
        result: Phase.Result
            A result object comprising the best fit model and other ci_data.
        """

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        result = self.optimizer.fit(analysis)

        return self.make_result(result=result, analysis=analysis)

    # noinspection PyMethodMayBeStatic
    def extract_ci_data(self, data, mask):
        return ci_data.MaskedCIData(image=data.image, noise_map=data.noise_map, ci_pre_cti=data.ci_pre_cti, mask=mask,
                                    ci_pattern=data.ci_pattern, ci_frame=data.ci_frame)

    def make_analysis(self, ci_datas, cti_settings, previous_results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        cti_settings
        ci_datas
        pool
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        masks = list(map(lambda data: self.mask_function(shape=data.image.shape), ci_datas))
        ci_datas_fit = [self.extract_ci_data(data=data, mask=mask) for data, mask in zip(ci_datas, masks)]
        ci_datas_full = list(map(lambda data, mask:
                                 ci_data.MaskedCIData(image=data.image, noise_map=data.noise_map,
                                                      ci_pre_cti=data.ci_pre_cti, mask=mask,
                                                      ci_pattern=data.ci_pattern, ci_frame=data.ci_frame),
                                 ci_datas, masks))

        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(ci_datas_extracted=ci_datas_fit, ci_datas_full=ci_datas_full,
                                           cti_settings=cti_settings, phase_name=self.phase_name,
                                           previous_results=previous_results, pool=pool)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(nl.Analysis):

        def __init__(self, ci_datas_extracted, ci_datas_full, cti_settings, phase_name, previous_results=None,
                     pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            self.ci_datas_extracted = ci_datas_extracted
            self.ci_datas_full = ci_datas_full
            self.cti_settings = cti_settings
            self.phase_name = phase_name
            self.phase_output_path = "{}/{}".format(conf.instance.output_path, self.phase_name)

            self.pool = pool or ConsecutivePool
            self.previous_results = previous_results

            self.plot_count = 0

            def output_bool(name):
                return conf.instance.general.get('output', name, bool)

            self.extract_array_from_mask = output_bool('extract_images_from_mask')

            self.plot_ci_data_as_subplot = output_bool('plot_ci_data_as_subplot')
            self.plot_ci_data_image = output_bool('plot_ci_data_image')
            self.plot_ci_data_noise_map = output_bool('plot_ci_data_noise_map')
            self.plot_ci_data_ci_pre_cti = output_bool('plot_ci_data_ci_pre_cti')
            self.plot_ci_data_signal_to_noise_map = output_bool('plot_ci_data_signal_to_noise_map')

            self.plot_ci_fit_all_at_end_png = output_bool('plot_ci_fit_all_at_end_png')
            self.plot_ci_fit_all_at_end_fits = output_bool('plot_ci_fit_all_at_end_fits')

            self.plot_ci_fit_as_subplot = output_bool('plot_ci_fit_as_subplot')
            self.plot_ci_fit_image = output_bool('plot_ci_fit_image')
            self.plot_ci_fit_noise_map = output_bool('plot_ci_fit_noise_map')
            self.plot_ci_fit_signal_to_noise_map = output_bool('plot_ci_fit_signal_to_noise_map')
            self.plot_ci_fit_ci_pre_cti = output_bool('plot_ci_fit_ci_pre_cti')
            self.plot_ci_fit_ci_post_cti = output_bool('plot_ci_fit_ci_post_cti')
            self.plot_ci_fit_residual_map = output_bool('plot_ci_fit_residual_map')
            self.plot_ci_fit_chi_squared_map = output_bool('plot_ci_fit_chi_squared_map')

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def visualize(self, instance, image_path, during_analysis):

            self.plot_count += 1

            ci_data_index = 0

            if self.plot_ci_data_as_subplot:
                ci_data_plotters.plot_ci_subplot(
                    ci_data=self.ci_datas_extracted[ci_data_index],
                    extract_array_from_mask=self.extract_array_from_mask,
                    output_path=image_path, output_format='png')

            ci_data_plotters.plot_ci_data_individual(
                ci_data=self.ci_datas_extracted[ci_data_index], extract_array_from_mask=self.extract_array_from_mask,
                should_plot_image=self.plot_ci_data_image,
                should_plot_noise_map=self.plot_ci_data_noise_map,
                should_plot_ci_pre_cti=self.plot_ci_data_ci_pre_cti,
                should_plot_signal_to_noise_map=self.plot_ci_data_signal_to_noise_map,
                output_path=image_path, output_format='png')

            fits = self.fits_of_ci_data_extracted_for_instance(instance=instance)

            ci_fit_index = 0

            if self.plot_ci_fit_as_subplot:
                ci_fit_plotters.plot_fit_subplot(
                    fit=fits[ci_fit_index], extract_array_from_mask=self.extract_array_from_mask,
                    output_path=image_path, output_format='png')

            if during_analysis:
                ci_fit_plotters.plot_fit_individuals(
                    fit=fits[ci_fit_index],
                    extract_array_from_mask=self.extract_array_from_mask,
                    should_plot_image=self.plot_ci_fit_image,
                    should_plot_noise_map=self.plot_ci_fit_noise_map,
                    should_plot_signal_to_noise_map=self.plot_ci_fit_signal_to_noise_map,
                    should_plot_ci_pre_cti=self.plot_ci_data_ci_pre_cti,
                    should_plot_ci_post_cti=self.plot_ci_fit_ci_post_cti,
                    should_plot_residual_map=self.plot_ci_fit_residual_map,
                    should_plot_chi_squared_map=self.plot_ci_fit_chi_squared_map,
                    output_path=image_path, output_format='png')

            elif not during_analysis:

                if self.plot_ci_fit_all_at_end_png:
                    ci_fit_plotters.plot_fit_individuals(
                        fit=fits[ci_fit_index],
                        extract_array_from_mask=self.extract_array_from_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_signal_to_noise_map=True,
                        should_plot_ci_pre_cti=True,
                        should_plot_ci_post_cti=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        output_path=image_path, output_format='png')

                if self.plot_ci_fit_all_at_end_fits:
                    ci_fit_plotters.plot_fit_individuals(
                        fit=fits[ci_fit_index],
                        extract_array_from_mask=self.extract_array_from_mask,
                        should_plot_image=True,
                        should_plot_noise_map=True,
                        should_plot_signal_to_noise_map=True,
                        should_plot_ci_pre_cti=True,
                        should_plot_ci_post_cti=True,
                        should_plot_residual_map=True,
                        should_plot_chi_squared_map=True,
                        output_path="{}/fits/".format(image_path), output_format='fits')

            return fits

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
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            likelihood = np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_extracted)))
            return likelihood

        def check_trap_lifetimes_are_ascending(self, cti_params):

            if hasattr(cti_params, 'serial'):

                if len(cti_params.serial) == 2:

                    if cti_params.serial_species[0].trap_lifetime > cti_params.serial_species[1].trap_lifetime:
                        raise exc.PriorException

                elif len(cti_params.serial) == 3:

                    if cti_params.serial_species[0].trap_lifetime > cti_params.serial_species[1].trap_lifetime or \
                            cti_params.serial_species[0].trap_lifetime > cti_params.serial_species[2].trap_lifetime or \
                            cti_params.serial_species[1].trap_lifetime > cti_params.serial_species[2].trap_lifetime:
                        raise exc.PriorException

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

    class Result(nl.Result):

        # noinspection PyUnusedLocal
        def __init__(self, constant, figure_of_merit, variable, analysis, optimizer):
            """
            The result of a phase
            """

            super(Phase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit, variable=variable)

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
    parallel_species = phase_property.PhasePropertyCollection("parallel_species")
    parallel_ccd = phase_property.PhaseProperty("parallel_ccd")

    def __init__(self, parallel_species=(), parallel_ccd=None, optimizer_class=nl.MultiNest,
                 mask_function=msk.Mask.empty_for_shape, columns=None,
                 phase_name="parallel_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(optimizer_class=optimizer_class, mask_function=mask_function, columns=columns, rows=None,
                         phase_name=phase_name)
        self.parallel_species = parallel_species
        self.parallel_ccd = parallel_ccd

    def extract_ci_data(self, data, mask):
        return data.parallel_calibration_data(
            columns=(0, self.columns or data.ci_frame.frame_geometry.parallel_overscan.total_columns),
            mask=mask)

    class Analysis(Phase.Analysis):

        def check_trap_lifetimes_are_ascending(self, cti_params):
            trap_lifetimes = [parallel_species.trap_lifetime for parallel_species in cti_params.parallel_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

        def describe(self, instance):
            return ("\nRunning CTI analysis for... \n\n"
                    "Parallel CTI: \n"
                    "Parallel Species:\n{}\n\n"
                    "Parallel CCD\n{}\n\n".format(instance.parallel_species, instance.parallel_ccd))


class SerialPhase(Phase):
    serial_species = phase_property.PhasePropertyCollection("serial_species")
    serial_ccd = phase_property.PhaseProperty("serial_ccd")

    def __init__(self, serial_species=(), serial_ccd=None, optimizer_class=nl.MultiNest,
                 mask_function=msk.Mask.empty_for_shape, rows=None, phase_name="serial_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(optimizer_class=optimizer_class, mask_function=mask_function, rows=rows, phase_name=phase_name)
        self.serial_species = serial_species
        self.serial_ccd = serial_ccd

    def extract_ci_data(self, data, mask):
        return data.serial_calibration_data(rows=self.rows or (0, data.ci_pattern.regions[0].total_rows),
                                            mask=mask)

    class Analysis(Phase.Analysis):

        def check_trap_lifetimes_are_ascending(self, cti_params):
            trap_lifetimes = [serial_species.trap_lifetime for serial_species in cti_params.serial_species]

            if not sorted(trap_lifetimes) == trap_lifetimes:
                raise exc.PriorException

        def describe(self, instance):
            return (
                "\nRunning CTI analysis for... \n\n"
                "Serial CTI: \n"
                "Serial Species:\n{}\n\n"
                "Serial CCD\n{}\n\n".format(instance.serial_species, instance.serial_ccd))


class ParallelSerialPhase(Phase):
    parallel_species = phase_property.PhasePropertyCollection("parallel_species")
    serial_species = phase_property.PhasePropertyCollection("serial_species")
    parallel_ccd = phase_property.PhaseProperty("parallel_ccd")
    serial_ccd = phase_property.PhaseProperty("serial_ccd")

    def __init__(self, parallel_species=(), serial_species=(), parallel_ccd=None, serial_ccd=None,
                 optimizer_class=nl.MultiNest, mask_function=msk.Mask.empty_for_shape,
                 phase_name="parallel_serial_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """
        super().__init__(optimizer_class=optimizer_class, mask_function=mask_function, columns=None,
                         rows=None, phase_name=phase_name)
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
    hyper_noise_scalars = phase_property.PhasePropertyCollection("hyper_noise_scalars")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        raise NotImplementedError()

    def noise_scaling_maps_from_result(self, result):
        raise NotImplementedError()

    def make_analysis(self, ci_datas, cti_settings, previous_results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        cti_settings
        ci_datas
        pool
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """
        masks = list(map(lambda data: self.mask_function(shape=data.image.shape), ci_datas))
        noise_scaling_maps = self.noise_scaling_maps_from_result(previous_results[-1])
        ci_datas_fit = [self.extract_ci_hyper_data(data=data, mask=mask, noise_scaling_maps=maps) for
                        data, mask, maps in zip(ci_datas, masks, noise_scaling_maps)]
        ci_datas_full = list(map(lambda data, mask, maps:
                                 ci_data.MaskedCIHyperData(image=data.image, noise_map=data.noise_map,
                                                           ci_pre_cti=data.ci_pre_cti, mask=mask,
                                                           ci_pattern=data.ci_pattern, ci_frame=data.ci_frame,
                                                           noise_scaling_maps=maps),
                                 ci_datas, masks, noise_scaling_maps))

        self.pass_priors(previous_results)

        class HyperAnalysis(self.__class__.Analysis):

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
                                 previous_results=previous_results, pool=pool)
        return analysis

    @property
    def number_of_noise_scalars(self):
        return len(self.hyper_noise_scalars)

    @number_of_noise_scalars.setter
    def number_of_noise_scalars(self, number):
        self.hyper_noise_scalars = [pm.PriorModel(ci_hyper.CIHyperNoiseScalar) for _ in range(number)]


class ParallelHyperPhase(ParallelPhase, HyperPhase):
    def __init__(self, parallel_species=(), parallel_ccd=None,
                 optimizer_class=nl.MultiNest, mask_function=msk.Mask.empty_for_shape, columns=None,
                 phase_name="parallel_hyper_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """
        super().__init__(parallel_species=parallel_species, parallel_ccd=parallel_ccd, optimizer_class=optimizer_class,
                         mask_function=mask_function, columns=columns, phase_name=phase_name)
        self.number_of_noise_scalars = 2

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """
        return [result.noise_scaling_maps_of_ci_regions, result.noise_scaling_maps_of_parallel_trails]

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.parallel_hyper_calibration_data(
            columns=(0, self.columns or data.ci_frame.frame_geometry.parallel_overscan.total_columns),
            mask=mask, noise_scaling_maps=noise_scaling_maps)


class SerialHyperPhase(SerialPhase, HyperPhase):

    def __init__(self, serial_species=(), serial_ccd=None,
                 optimizer_class=nl.MultiNest, mask_function=msk.Mask.empty_for_shape, rows=None,
                 phase_name="serial_hyper_phase"):
        """
        A phase with a simple source/CTI model
        """
        HyperPhase.__init__(self=self)
        super().__init__(serial_species=serial_species, serial_ccd=serial_ccd, optimizer_class=optimizer_class,
                         mask_function=mask_function, rows=rows, phase_name=phase_name)
        self.number_of_noise_scalars = 2

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """
        return [result.noise_scaling_maps_of_ci_regions, result.noise_scaling_maps_of_serial_trails]

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.serial_hyper_calibration_data(rows=self.rows or (0, data.ci_pattern.regions[0].total_rows),
                                                  mask=mask, noise_scaling_maps=noise_scaling_maps)


class ParallelSerialHyperPhase(ParallelSerialPhase, HyperPhase):
    def __init__(self, parallel_species=(), serial_species=(), parallel_ccd=None, serial_ccd=None,
                 optimizer_class=nl.MultiNest, mask_function=msk.Mask.empty_for_shape,
                 phase_name="parallel_serial_hyper_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        optimizer_class: class
            The class of a non-linear optimizer
        """
        super().__init__(parallel_species=parallel_species, serial_species=serial_species, parallel_ccd=parallel_ccd,
                         serial_ccd=serial_ccd, optimizer_class=optimizer_class, mask_function=mask_function,
                         phase_name=phase_name)
        self.number_of_noise_scalars = 4

    def noise_scaling_maps_from_result(self, result):
        """
        Extract relevant noise scalings maps from the previous result.
        """
        return [result.noise_scaling_maps_of_ci_regions,
                result.noise_scaling_maps_of_parallel_trails,
                result.noise_scaling_maps_of_serial_trails,
                result.noise_scaling_maps_of_serial_overscan_above_trails]

    def extract_ci_hyper_data(self, data, mask, noise_scaling_maps):
        return data.parallel_serial_hyper_calibration_data(mask, noise_scaling_maps=noise_scaling_maps)


def pipe_cti(ci_data_fit, cti_params, cti_settings):
    fit = ci_fit.CIFit(masked_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings)
    return fit.figure_of_merit


def pipe_cti_hyper(ci_data_fit, cti_params, cti_settings, hyper_noise_scalars):
    fit = ci_fit.CIHyperFit(masked_hyper_ci_data=ci_data_fit, cti_params=cti_params, cti_settings=cti_settings,
                            hyper_noise_scalars=hyper_noise_scalars)
    return fit.figure_of_merit
