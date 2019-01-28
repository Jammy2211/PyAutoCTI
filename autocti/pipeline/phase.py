import inspect
import logging
import os
from functools import partial

import numpy as np
from astropy.io import fits
from autofit import conf
from autofit.tools import phase as ph
from autofit.tools import phase_property
from autofit.mapper import model_mapper as mm
from autofit.optimize import non_linear as nl

from autocti.data import util
from autocti.data import mask as msk
from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_hyper
from autocti.data.charge_injection.plotters import ci_plotters
from autocti.data.fitting import fitting
from autocti.model import arctic_params

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

def default_mask_function(image):
    return msk.Mask.empty_for_shape(shape=image.shape, frame_geometry=image.frame_geometry, ci_pattern=image.ci_pattern)

class HyperOnly(ph.AbstractPhase):
    pass


def cti_params_for_instance(instance):
    return arctic_params.ArcticParams(
        parallel_ccd=instance.parallel_ccd if hasattr(instance, "parallel_ccd") else None,
        parallel_species=instance.parallel_species if hasattr(instance, "parallel_species") else None,
        serial_ccd=instance.serial_ccd if hasattr(instance, "serial_ccd") else None,
        serial_species=instance.serial_species if hasattr(instance, "serial_species") else None
    )


class ConsecutivePool(object):
    @staticmethod
    def map(func, ls):
        return map(func, ls)


class Phase(ph.AbstractPhase):

    def __init__(self, optimizer_class=nl.DownhillSimplex, mask_function=default_mask_function, columns=None, rows=None,
                 phase_name=None):
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
        return self.__class__.Result(result.constant, result.likelihood, result.variable, analysis)

    # noinspection PyMethodMayBeStatic
    def extract_ci_data(self, ci_datas):
        return ci_datas

    def make_analysis(self, ci_datas, cti_settings, masks=None, previous_results=None, pool=None):
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

        if masks is None:
            masks = list(map(lambda ci_data : self.mask_function(image=ci_data.image), ci_datas))

        # TODO : Make mask an input of extract_ci_data and use the class method to edit in the mask.
        # TODO : ci_datas_fit should include the mask as an attrbiute.

        ci_datas_fit = self.extract_ci_data(ci_datas=ci_datas)

        ci_datas_fit[0].mask = masks[0]

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

            fitter = self.fit_for_instance(instance)
            ci_post_ctis = fitter.ci_post_ctis
            residuals = fitter.residuals
            chi_squareds = fitter.chi_squareds

            for i in range(len(ci_post_ctis)):
                self.output_array_as_fits(ci_post_ctis[i], "ci_post_ctis_" + str(i))
                self.output_array_as_fits(residuals[i], "residuals_" + str(i))
                self.output_array_as_fits(chi_squareds[i], "chi_squareds_" + str(i))

            return fitter, ci_post_ctis, residuals, chi_squareds

        def output_array_as_fits(self, array, filename):

            file = self.output_image_path + filename + '.fits'

            if os.path.isfile(file):
                os.remove(file)

            try:
                if array is not None:
                    hdu = fits.PrimaryHDU()
                    hdu.data = array
                    hdu.writeto(file)
            except OSError as e:
                logger.exception(e)

        def fit(self, **kwargs):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            kwargs: dict
                Dictionary of objects describing the model

            Returns
            -------
            fit: fitting.Fit
                How fit the model is and the model
            """
            raise NotImplementedError()

        @classmethod
        def log(cls, instance):
            raise NotImplementedError()

        def fitter_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.CIFitter(self.ci_datas, cti_params=cti_params, cti_settings=self.cti_settings)

        def fit_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.CIFitter(self.ci_datas_fit, cti_params=cti_params, cti_settings=self.cti_settings)

    class Result(nl.Result):

        def __init__(self, constant, likelihood, variable, *args):
            """
            The result of a phase
            """
            super(Phase.Result, self).__init__(constant, likelihood, variable)

    def pass_priors(self, previous_results):
        """
        Perform any prior or constant passing. This could involve setting model attributes equal to priors or constants
        from a previous phase.

        Parameters
        ----------
        previous_results: ResultsCollection
            The result of the previous phase
        """
        pass


def is_prior(value):
    return inspect.isclass(value) or isinstance(value, mm.AbstractPriorModel)


class ParallelPhase(Phase):

    parallel_species = phase_property.PhasePropertyCollection("parallel_species")
    parallel_ccd = phase_property.PhaseProperty("parallel_ccd")

    def __init__(self, parallel_species=(), parallel_ccd=None, optimizer_class=nl.MultiNest,
                 mask_function=default_mask_function, columns=None,
                 phase_name="parallel_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(optimizer_class=optimizer_class, mask_function=mask_function, columns=columns, rows=None,
                         phase_name=phase_name)
        self.parallel_species = parallel_species
        self.parallel_ccd = parallel_ccd

    def extract_ci_data(self, ci_datas):
        return [data.parallel_calibration_data(
            (0, self.columns or data.image.frame_geometry.parallel_overscan.total_columns)) for data in ci_datas]

    class Analysis(Phase.Analysis):

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
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            if self.pool is not None:
                return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))
            fitter = fitting.CIFitter(self.ci_datas_fit, cti_params=cti_params, cti_settings=self.cti_settings)
            return fitter.likelihood

        def visualize(self, instance, suffix, during_analysis):

            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)

            return fitter, ci_post_ctis, residuals, chi_squareds

        def output_ci_regions_binned_across_serial(self, images, masks, filename):

            for i in range(len(images)):
                ci_plotters.ci_regions_binned_across_serial(images[i], masks[i], path=self.output_image_path,
                                                            filename=filename + str(i), line0=False)

        def output_parallel_trails_binned_across_serial(self, images, masks, filename):

            for i in range(len(images)):
                ci_plotters.parallel_trails_binned_across_serial(images[i], masks[i], path=self.output_image_path,
                                                                 filename=filename + str(i), line0=False)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\nParallel CTI::\n{}\n\n".format(instance.parallel))

        def noise_scalings_for_instance(self, instance):
            """
            First noises scaling images are of the charge injection regions.
            Second noises scaling images are of the non-charge injection regions in the parallel calibration ci_frame"""
            fitter = self.fitter_for_instance(instance)
            return list(map(lambda chi_squared:
                            [chi_squared.ci_regions_frame_from_frame(),
                             chi_squared.parallel_non_ci_regions_frame_from_frame()],
                            fitter.chi_squareds))

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            super().__init__(constant, likelihood, variable, analysis)
            self.noise_scalings = analysis.noise_scalings_for_instance(constant)


class ParallelHyperPhase(ParallelPhase):

    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = phase_property.PhaseProperty("hyp_parallel_trails")

    def __init__(self, parallel_species=(), parallel_ccd=None, hyp_ci_regions=None, hyp_parallel_trails=None,
                 optimizer_class=nl.MultiNest, mask_function=default_mask_function, columns=None,
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

    class Analysis(ParallelPhase.Analysis):
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
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_parallel_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            pass

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI::\n{}\n\n "
                "Hyper Parameters:\n{}\n{}\n".format(instance.parallel, instance.hyp_ci_regions,
                                                     instance.hyp_parallel_trails))

        def fitter_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas, cti_params=cti_params, cti_settings=self.cti_settings,
                                         hyper_noises=[instance.hyp_ci_regions,
                                                       instance.hyp_parallel_trails])

        def fit_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas_fit, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                                                       instance.hyp_parallel_trails])


class ParallelHyperOnlyPhase(ParallelHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class ParallelHyper(ParallelHyperPhase):
            # noinspection PyShadowingNames
            def pass_priors(self, previous_results):
                self.parallel_species = previous_results.last.constant.parallel_species
                self.parallel_ccd = previous_results.last.constant.parallel_ccd

        phase = ParallelHyper(optimizer_class=nl.MultiNest, hyp_ci_regions=ci_hyper.HyperCINoise,
                              hyp_parallel_trails=ci_hyper.HyperCINoise,
                              phase_name=self.phase_name)

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        hyper_result = phase.run(ci_datas, cti_settings, previous_results, pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.figure_of_merit, hyper_result.variable,
                                     analysis)


class SerialPhase(Phase):
    serial_species = phase_property.PhasePropertyCollection("serial_species")
    serial_ccd = phase_property.PhaseProperty("serial_ccd")

    def __init__(self, serial_species=(), serial_ccd=None, optimizer_class=nl.MultiNest,
                 mask_function=default_mask_function, columns=None, rows=None, phase_name="serial_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(optimizer_class=optimizer_class, mask_function=mask_function, columns=columns,
                         rows=rows, phase_name=phase_name)
        self.serial_species = serial_species
        self.serial_ccd = serial_ccd

    def extract_ci_data(self, ci_datas):
        columns = self.columns or 0
        return [data.serial_calibration_data(
            columns or 0, self.rows or (0, data.image.ci_pattern.regions[0].total_rows)) for data in ci_datas]

    class Analysis(Phase.Analysis):
        def fit(self, instance):
            """
            Runs the analysis. Determine how well the supplied cti_params fits the image.

            Params
            ----------
            serial : arctic_params.SerialParams
                A class describing the serial cti model and parameters.
            serial : arctic_params.SerialParams
                A class describing the serial cti model and parameters.
            hyp : ci_hyper.HyperCINoise
                A class describing the noises scaling ci_hyper-parameters.

            Returns
            -------
            result: Result
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))

        def visualize(self, instance, suffix, during_analysis):
            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)
            return fitter, ci_post_ctis, residuals, chi_squareds

        def output_ci_regions_binned_across_parallel(self, images, masks, filename):
            for i in range(len(images)):
                ci_plotters.ci_regions_binned_across_parallel(images[i], masks[i], path=self.output_image_path,
                                                              filename=filename + str(i), line0=False)

        def output_serial_trails_binned_across_parallel(self, images, masks, filename):
            for i in range(len(images)):
                ci_plotters.serial_trails_binned_across_parallel(images[i], masks[i], path=self.output_image_path,
                                                                 filename=filename + str(i), line0=False)

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\nSerial CTI::\n{}\n\n".format(instance.serial))

        def noise_scalings_for_instance(self, instance):
            """
            First noises scaling images are of the charge injection regions.
            Second noises scaling images are of the non-charge injection regions in the serial calibration ci_frame
            """
            fitter = self.fitter_for_instance(instance)
            return list(map(lambda chi_squared:
                            [chi_squared.ci_regions_frame_from_frame(),
                             chi_squared.serial_all_trails_frame_from_frame()],
                            fitter.chi_squareds))

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            super().__init__(constant, likelihood, variable, analysis)
            self.noise_scalings = analysis.noise_scalings_for_instance(constant)


class SerialHyperPhase(SerialPhase):
    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_serial_trails = phase_property.PhaseProperty("hyp_serial_trails")

    def __init__(self, serial_species=(), serial_ccd=None, hyp_ci_regions=None, hyp_serial_trails=None,
                 optimizer_class=nl.MultiNest, mask_function=default_mask_function, columns=None, rows=None,
                 phase_name="serial_hyper_phase"):
        """
        A phase with a simple source/CTI model
        """
        super().__init__(serial_species=serial_species, serial_ccd=serial_ccd, optimizer_class=optimizer_class,
                         mask_function=mask_function, columns=columns, rows=rows, phase_name=phase_name)
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_serial_trails = hyp_serial_trails
        self.has_noise_scalings = True

    class Analysis(SerialPhase.Analysis):

        def fit(self, instance):
            """
            Runs the analysis. Determine how well the supplied cti_params fits the image.

            Params
            ----------
            serial : arctic_params.SerialParams
                A class describing the serial cti model and parameters.
            serial : arctic_params.SerialParams
                A class describing the serial cti model and parameters.
            hyp : ci_hyper.HyperCINoise
                A class describing the noises scaling ci_hyper-parameters.

            Returns
            -------
            result: Result
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_serial_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))

        def visualize(self, instance, suffix, during_analysis):
            pass

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Serial CTI::\n{}\n\n "
                "Hyper Parameters:\n{}\n{}\n".format(instance.serial, instance.hyp_ci_regions,
                                                     instance.hyp_serial_trails))

        def hyper_fit_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas_fit, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                                                       instance.hyp_serial_trails])


class SerialHyperOnlyPhase(SerialHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class SerialHyper(SerialHyperPhase):
            # noinspection PyShadowingNames
            def pass_priors(self, previous_results):
                self.serial_species = previous_results.last.constant.serial_species
                self.serial_ccd = previous_results.last.constant.serial_ccd

        phase = SerialHyper(optimizer_class=nl.MultiNest, hyp_ci_regions=ci_hyper.HyperCINoise,
                            hyp_serial_trails=ci_hyper.HyperCINoise,
                            phase_name=self.phase_name)

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        hyper_result = phase.run(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                 pool=pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.figure_of_merit, hyper_result.variable,
                                     analysis)


class ParallelSerialPhase(Phase):
    parallel_species = phase_property.PhasePropertyCollection("parallel_species")
    serial_species = phase_property.PhasePropertyCollection("serial_species")
    parallel_ccd = phase_property.PhaseProperty("parallel_ccd")
    serial_ccd = phase_property.PhaseProperty("serial_ccd")

    def __init__(self, parallel_species=(), serial_species=(), parallel_ccd=None, serial_ccd=None,
                 optimizer_class=nl.MultiNest, mask_function=default_mask_function,
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

    def extract_ci_data(self, ci_datas):
        return [data.parallel_serial_calibration_data() for data in ci_datas]

    class Analysis(Phase.Analysis):
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
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance=instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_datas_fit)))

        def visualize(self, instance, suffix, during_analysis):
            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)

            return fitter, ci_post_ctis, residuals, chi_squareds

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... "
                "\n\nParallel CTI::\n{}"
                "\n\nSerial CTI::\n{}\n\n".format(instance.parallel, instance.serial))

        def noise_scalings_for_instance(self, instance):
            """

            First noises scaling images are of the charge injection regions.
            Second noises scaling images are of the non-charge injection regions in the parallel calibration ci_frame"""
            cti_params = cti_params_for_instance(instance)
            fitter = fitting.CIFitter(self.ci_datas_fit, cti_params=cti_params, cti_settings=self.cti_settings)
            return list(map(lambda chi_squared:
                            [chi_squared.ci_regions_frame_from_frame(),
                             chi_squared.parallel_non_ci_regions_frame_from_frame(),
                             chi_squared.serial_all_trails_frame_from_frame(),
                             chi_squared.serial_overscan_non_trails_frame_from_frame()],
                            fitter.chi_squareds))

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            super().__init__(constant, likelihood, variable, analysis)
            self.noise_scalings = analysis.noise_scalings_for_instance(constant)


class ParallelSerialHyperPhase(ParallelSerialPhase):
    hyp_ci_regions = phase_property.PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = phase_property.PhaseProperty("hyp_parallel_trails")
    hyp_serial_trails = phase_property.PhaseProperty("hyp_serial_trails")
    hyp_parallel_serial_trails = phase_property.PhaseProperty("hyp_parallel_serial_trails")

    def __init__(self, parallel_species=(), serial_species=(), parallel_ccd=None, serial_ccd=None, hyp_ci_regions=None,
                 hyp_parallel_trails=None, hyp_serial_trails=None, hyp_parallel_serial_trails=None,
                 optimizer_class=nl.MultiNest, mask_function=default_mask_function,
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

    class Analysis(ParallelPhase.Analysis):
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
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            cti_params = cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_parallel_trails,
                                                  instance.hyp_serial_trails,
                                                  instance.hyp_parallel_serial_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            pass

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI::\n{}\n\n "
                "Serial CTI::\n{}\n\n "
                "Hyper Parameters:\n{}\n{}\n{}\n{}\n".format(instance.parallel, instance.serial,
                                                             instance.hyp_ci_regions, instance.hyp_parallel_trails,
                                                             instance.hyp_serial_trails,
                                                             instance.hyp_parallel_serial_trails))

        def fit_for_instance(self, instance):
            cti_params = cti_params_for_instance(instance)
            return fitting.HyperCIFitter(ci_datas_fit=self.ci_datas_fit, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                       instance.hyp_parallel_trails,
                                                       instance.hyp_serial_trails,
                                                       instance.hyp_parallel_serial_trails])


class ParallelSerialHyperOnlyPhase(ParallelSerialHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class ParallelSerialHyper(ParallelSerialHyperPhase):
            # noinspection PyShadowingNames
            def pass_priors(self, previous_results):
                self.serial_species = previous_results[-1].constant.serial_species
                self.serial_ccd = previous_results[-1].constant.serial_ccd
                self.parallel_species = previous_results[-1].constant.parallel_species
                self.parallel_ccd = previous_results[-1].constant.parallel_ccd

        phase = ParallelSerialHyper(optimizer_class=nl.MultiNest,
                                    hyp_ci_regions=ci_hyper.HyperCINoise,
                                    hyp_parallel_trails=ci_hyper.HyperCINoise,
                                    hyp_serial_trails=ci_hyper.HyperCINoise,
                                    hyp_parallel_serial_trails=ci_hyper.HyperCINoise,
                                    phase_name=self.phase_name)

        phase.optimizer.n_live_points = 50
        phase.optimizer.sampling_efficiency = 0.5

        hyper_result = phase.run(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                 pool=pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.figure_of_merit, hyper_result.variable,
                                     analysis)


def pipe_cti(ci_pipe_data, cti_params, cti_settings):
    fitter = fitting.CIFitter(ci_datas_fit=[ci_pipe_data], cti_params=cti_params, cti_settings=cti_settings)
    return fitter.likelihood


def pipe_cti_hyper(ci_pipe_data, cti_params, cti_settings, hyper_noises):
    fitter = fitting.HyperCIFitter(ci_datas_fit=[ci_pipe_data], cti_params=cti_params, cti_settings=cti_settings,
                                   hyper_noises=hyper_noises)
    return fitter.scaled_likelihood
