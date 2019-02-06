import logging
from functools import partial

import numpy as np
from autofit import conf
from autofit.optimize import non_linear as nl
from autofit.tools import phase as ph
from autofit.tools import phase_property

from autocti.charge_injection import ci_hyper, ci_fit, ci_data
from autocti.data import mask as msk
from autocti.data import util
from autocti.model import arctic_params

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


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
        return self.__class__.Result(result.constant, result.figure_of_merit, result.variable, analysis)

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
        result: nl.Result
            A result object comprising the best fit model and other ci_data.
        """

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        result = self.optimizer.fit(analysis)
        if analysis.visualize_results:
            analysis.visualize(instance=result.constant, suffix=None, during_analysis=False)
        return self.make_result(result, analysis)

    # noinspection PyMethodMayBeStatic
    def extract_ci_data(self, data, mask):
        return ci_data.CIDataFit(image=data.image, noise_map=data.noise_map, ci_pre_cti=data.ci_pre_cti, mask=mask,
                                 ci_pattern=data.ci_pattern, ci_frame=data.ci_frame,noise_scaling=data.noise_scaling)

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

        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(ci_datas_fit=ci_datas_fit,
                                           cti_settings=cti_settings, phase_name=self.phase_name,
                                           previous_results=previous_results, pool=pool)
        return analysis

    # noinspection PyAbstractClass
    class Analysis(nl.Analysis):

        def __init__(self, ci_datas_fit, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a \
            set of objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """

            self.ci_datas_fit = ci_datas_fit
            self.cti_settings = cti_settings
            self.phase_name = phase_name
            self.pool = pool or ConsecutivePool
            self.previous_results = previous_results

            self.visualize_results = conf.instance.general.get('output', 'visualize_results', bool)

            self.plot_count = 0
            self.output_image_path = "{}/".format(conf.instance.output_path) + '/' + self.phase_name + '/images/'
            util.make_path_if_does_not_exist(self.output_image_path)

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def visualize(self, instance, suffix, during_analysis):
            pass

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
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))

        @classmethod
        def log(cls, instance):
            raise NotImplementedError()

        def fit_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return ci_fit.CIFit(ci_data_fit=self.ci_datas_fit, cti_params=cti_params, cti_settings=self.cti_settings)

    class Result(nl.Result):

        # noinspection PyUnusedLocal
        def __init__(self, constant, figure_of_merit, variable, *args):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant=constant, figure_of_merit=figure_of_merit, variable=variable)


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
            (0, self.columns or data.ci_frame.parallel_overscan.total_columns), mask)


    class Analysis(Phase.Analysis):
        
        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI: \n"
                "Parallel Species:\n{}\n\n "
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
        columns = self.columns or 0
        return data.serial_calibration_data(columns or 0, self.rows or (0, data.ci_pattern.regions[0].total_rows),
                                            mask)

    class Analysis(Phase.Analysis):
        
        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... "
                "\n\nSerial CTI: \n"
                "Serial Species:\n{}\n\n "
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

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI: \n"
                "Parallel Species:\n{}\n\n "
                "Parallel CCD\n{}\n\n"
                "Serial CTI: \n"
                "Serial Species:\n{}\n\n "
                "Serial CCD\n{}\n\n".format(instance.parallel_species, instance.parallel_ccd,
                                              instance.serial_species, instance.serial_ccd))


class HyperAnalysis(Phase.Analysis):

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
        cti_params = cti_params_for_instance(instance)
        pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                hyper_noises=self.noises_from_instance(instance))
        return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))

    @classmethod
    def log(cls, instance):
        logger.debug(
            "\nRunning CTI analysis for... \n\n"
            "Parallel CTI::\n{}\n\n "
            "Hyper Parameters:\n{}".format(instance.parallel, " ".join(cls.noises_from_instance(instance))))

    def fit_for_instance(self, instance):
        cti_params = cti_params_for_instance(instance)
        return ci_fit.CIHyperFit(ci_data_fit=self.ci_datas_fit,
                                 cti_params=cti_params,
                                 cti_settings=self.cti_settings,
                                 hyper_noises=self.noises_from_instance(instance))

    @classmethod
    def noises_from_instance(cls, instance):
        raise NotImplementedError()


class ParallelHyperPhase(ParallelPhase):

    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = phase_property.PhaseProperty("hyp_parallel_trails")

    def __init__(self, parallel_species=(), parallel_ccd=None, hyp_ci_regions=None, hyp_parallel_trails=None,
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

        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_parallel_trails = hyp_parallel_trails
        self.has_noise_scalings = True

    class Analysis(HyperAnalysis, ParallelPhase.Analysis):

        @classmethod
        def noises_from_instance(cls, instance):
            return [instance.hyp_ci_regions, instance.hyp_parallel_trails]


class SerialHyperPhase(SerialPhase):

    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_serial_trails = phase_property.PhaseProperty("hyp_serial_trails")

    def __init__(self, serial_species=(), serial_ccd=None, hyp_ci_regions=None, hyp_serial_trails=None,
                 optimizer_class=nl.MultiNest, mask_function=msk.Mask.empty_for_shape, rows=None,
                 phase_name="serial_hyper_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(serial_species=serial_species, serial_ccd=serial_ccd, optimizer_class=optimizer_class,
                         mask_function=mask_function, columns=columns, rows=rows, phase_name=phase_name)
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_serial_trails = hyp_serial_trails
        self.has_noise_scalings = True

    class Analysis(HyperAnalysis, SerialPhase.Analysis):
        @classmethod
        def noises_from_instance(cls, instance):
            return [instance.hyp_ci_regions, instance.hyp_serial_trails]


class ParallelSerialHyperPhase(ParallelSerialPhase):
    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = phase_property.PhaseProperty("hyp_parallel_trails")
    hyp_serial_trails = phase_property.PhaseProperty("hyp_serial_trails")
    hyp_parallel_serial_trails = phase_property.PhaseProperty("hyp_parallel_serial_trails")

    def __init__(self, parallel_species=(), serial_species=(), parallel_ccd=None, serial_ccd=None, hyp_ci_regions=None,
                 hyp_parallel_trails=None, hyp_serial_trails=None, hyp_parallel_serial_trails=None,
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
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_parallel_trails = hyp_parallel_trails
        self.hyp_serial_trails = hyp_serial_trails
        self.hyp_parallel_serial_trails = hyp_parallel_serial_trails
        self.has_noise_scalings = True

    class Analysis(HyperAnalysis, ParallelPhase.Analysis):
        @classmethod
        def noises_from_instance(cls, instance):
            return [instance.hyp_ci_regions,
                    instance.hyp_parallel_trails,
                    instance.hyp_serial_trails,
                    instance.hyp_parallel_serial_trails]


def pipe_cti(ci_data_fit, cti_params, cti_settings):
    fitter = ci_fit.CIFit(ci_data_fit=ci_data_fit,
                          cti_params=cti_params,
                          cti_settings=cti_settings)
    return fitter.figure_of_merit


def pipe_cti_hyper(ci_data_fit, cti_params, cti_settings, hyper_noises):
    fitter = ci_fit.CIHyperFit(ci_data_fit=ci_data_fit,
                               cti_params=cti_params,
                               cti_settings=cti_settings,
                               hyper_noises=hyper_noises)
    return fitter.figure_of_merit
