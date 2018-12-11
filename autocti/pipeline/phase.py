import inspect
import logging
import os
from functools import partial

import numpy as np
from astropy.io import fits
from autofit import conf
from autofit.core import model_mapper as mm
from autofit.core import non_linear as nl

from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_hyper
from autocti.data.charge_injection.plotters import ci_plotters
from autocti.data.fitting import fitting
from autocti.model import arctic_params
from autocti.tools import imageio

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def default_extractor(ci_datas, mask_function, columns=None, rows=None, noise_scalings=None):
    images = list(map(lambda ci_data: ci_data.image, ci_datas))
    masks = list(map(lambda ci_data: mask_function(ci_data.mask), ci_datas))
    noises = list(map(lambda ci_data: ci_data.noise, ci_datas))
    ci_pre_ctis = list(map(lambda ci_data: ci_data.ci_pre_cti, ci_datas))

    new_noise_scalings = []

    if noise_scalings is not None:

        for i in range(len(noise_scalings)):
            new_noise_scalings.append(list(map(lambda noise_scaling: noise_scaling, noise_scalings[i])))

    return ci_data.CIDataAnalysis(images, masks, noises, ci_pre_ctis, new_noise_scalings)


def parallel_extractor(ci_datas, mask_function, columns=None, rows=None, noise_scalings=None):
    if columns is None:
        columns = ci_datas[0].image.cti_geometry.parallel_overscan.total_columns

    images = list(map(lambda ci_data:
                      ci_data.image.parallel_calibration_section_from_frame(columns=(0, columns)), ci_datas))

    masks_modify = list(map(lambda ci_data: mask_function(ci_data.mask), ci_datas))
    masks = list(map(lambda mask:
                     mask.parallel_calibration_section_from_frame(columns=(0, columns)), masks_modify))

    noises = list(map(lambda ci_data:
                      ci_data.noise.parallel_calibration_section_from_frame(columns=(0, columns)), ci_datas))

    ci_pre_ctis = list(map(lambda ci_data:
                           ci_data.ci_pre_cti.parallel_calibration_section_from_frame(columns=(0, columns)), ci_datas))

    new_noise_scalings = []

    if noise_scalings is not None:

        for i in range(len(noise_scalings)):
            new_noise_scalings.append(list(map(lambda noise_scaling:
                                               noise_scaling.parallel_calibration_section_from_frame(
                                                   columns=(0, columns)
                                               ),
                                               noise_scalings[i])))

    return ci_data.CIDataAnalysis(images, masks, noises, ci_pre_ctis, new_noise_scalings)


def serial_extractor(ci_datas, mask_function, columns=None, rows=None, noise_scalings=None):
    if columns is None:
        columns = ci_datas[0].image.cti_geometry.serial_prescan.x1

    columns = 0

    if rows is None:
        rows = (0, ci_datas[0].image.ci_pattern.regions[0].total_rows)
        rows = (0, 3)

    images = list(map(lambda ci_data:
                      ci_data.image.serial_calibration_array_from_frame(from_column=columns, rows=rows), ci_datas))

    masks_modify = list(map(lambda ci_data: mask_function(ci_data.mask), ci_datas))
    masks = list(map(lambda mask:
                     mask.serial_calibration_array_from_frame(from_column=columns, rows=rows), masks_modify))

    noises = list(map(lambda ci_data:
                      ci_data.noise.serial_calibration_array_from_frame(from_column=columns, rows=rows), ci_datas))

    ci_pre_ctis = list(map(lambda ci_data:
                           ci_data.ci_pre_cti.serial_calibration_array_from_frame(from_column=columns, rows=rows),
                           ci_datas))

    new_noise_scalings = []

    if noise_scalings is not None:

        for i in range(len(noise_scalings)):
            new_noise_scalings.append(list(map(lambda noise_scaling:
                                               noise_scaling.serial_calibration_array_from_frame(from_column=columns,
                                                                                                 rows=rows),
                                               noise_scalings[i])))

    return ci_data.CIDataAnalysis(images, masks, noises, ci_pre_ctis, new_noise_scalings)


def parallel_serial_extractor(ci_datas, mask_function, columns=None, rows=None, noise_scalings=None):
    images = list(map(lambda ci_data:
                      ci_data.image.parallel_serial_calibration_section_from_frame(), ci_datas))

    masks_modify = list(map(lambda ci_data: mask_function(ci_data.mask), ci_datas))
    masks = list(map(lambda mask:
                     mask.parallel_serial_calibration_section_from_frame(), masks_modify))

    noises = list(map(lambda ci_data:
                      ci_data.noise.parallel_serial_calibration_section_from_frame(), ci_datas))

    ci_pre_ctis = list(map(lambda ci_data:
                           ci_data.ci_pre_cti.parallel_serial_calibration_section_from_frame(), ci_datas))

    new_noise_scalings = []

    if noise_scalings is not None:

        for i in range(len(noise_scalings)):
            new_noise_scalings.append(list(map(lambda noise_scaling:
                                               noise_scaling.parallel_serial_calibration_section_from_frame(),
                                               noise_scalings[i])))

    return ci_data.CIDataAnalysis(images, masks, noises, ci_pre_ctis, new_noise_scalings)


def default_mask_function(mask):
    return mask


class ResultsCollection(list):
    def __init__(self, results):
        super().__init__(results)

    @property
    def last(self):
        if len(self) > 0:
            return self[-1]
        return None

    @property
    def first(self):
        if len(self) > 0:
            return self[0]
        return None


class IntervalCounter(object):
    def __init__(self, interval):
        self.count = 0
        self.interval = interval

    def __call__(self):
        self.count += 1
        return self.count % self.interval == 0


class HyperOnly(object):
    pass


class Phase(object):

    def __init__(self, optimizer_class=nl.DownhillSimplex, ci_datas_extractor=default_extractor,
                 columns=None, rows=None, mask_function=default_mask_function, phase_name=None):
        """
        A phase in an analysis pipeline. Uses the set NonLinear optimizer to try to fit models and images passed to it.

        Parameters
        ----------
        optimizer_class: class
            The class of a NonLinear optimizer
            The side length of the subgrid
        sub_grid_size: int
        """
        self.optimizer = optimizer_class(name=phase_name)
        self.ci_datas_extractor = ci_datas_extractor
        self.columns = columns
        self.rows = rows
        self.mask_function = mask_function
        self.phase_name = phase_name
        self.has_noise_scalings = False

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

    @property
    def doc(self):
        if self.__doc__ is not None:
            return self.__doc__.replace("  ", "").replace("\n", " ")

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        previous_results: ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        image: img.Image
            An image that has been masked

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

    def make_analysis(self, ci_datas, cti_settings, previous_results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        image: im.Image
            An image that has been masked
        previous_results: ResultsCollection
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """

        if not self.has_noise_scalings:
            noise_scalings = None
        elif self.has_noise_scalings:
            noise_scalings = previous_results.last.noise_scalings

        ci_datas_analysis = self.ci_datas_extractor(ci_datas, mask_function=self.mask_function, columns=self.columns,
                                                    rows=self.rows, noise_scalings=noise_scalings)
        self.pass_priors(previous_results)
        analysis = self.__class__.Analysis(ci_datas=ci_datas, ci_datas_analysis=ci_datas_analysis,
                                           cti_settings=cti_settings, phase_name=self.phase_name,
                                           previous_results=previous_results, pool=pool)
        return analysis

    class Analysis(object):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            self.ci_datas = ci_datas
            self.ci_datas_analysis = ci_datas_analysis
            if pool is not None:
                self.ci_pipe_data = [[self.ci_datas_analysis[i].image, self.ci_datas_analysis[i].mask,
                                      self.ci_datas_analysis[i].noise, self.ci_datas_analysis[i].ci_pre_cti,
                                      self.ci_datas_analysis[i].noise_scalings]
                                     for i in range(len(self.ci_datas_analysis))]

            self.cti_settings = cti_settings
            self.phase_name = phase_name
            self.pool = pool
            self.previous_results = previous_results
            log_interval = conf.instance.general.get('output', 'log_interval', int)
            self.visualize_results = conf.instance.general.get('output', 'visualize_results', bool)

            self.__should_log = IntervalCounter(log_interval)
            self.plot_count = 0
            self.output_image_path = "{}/".format(conf.instance.output_path) + '/' + self.phase_name + '/images/'
            imageio.make_path_if_does_not_exist(self.output_image_path)

        @property
        def should_log(self):
            return self.__should_log()

        @property
        def last_results(self):
            if self.previous_results is not None:
                return self.previous_results.last

        def visualize(self, instance, suffix, during_analysis):

            fitter = self.fitter_analysis_for_instance(instance)
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

        def try_log(self, instance):
            """
            Determine the fitness of a particular model

            Parameters
            ----------
            instance
                A model instance

            Returns
            -------
            fit: fitting.Fit
                How fit the model is and the model
            """
            if self.should_log:
                self.log(instance)
            return None

        def fit_pool(self, **kwargs):
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

        def cti_params_for_instance(self, instance):
            raise NotImplementedError()

        def fitter_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.CIFitter(self.ci_datas, cti_params=cti_params, cti_settings=self.cti_settings)

        def fitter_analysis_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.CIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings)

    class Result(nl.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase

            Parameters
            ----------

            galaxy_images: [ndarray]
                A collection of images created by each individual galaxy which taken together form the full model image
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


class PhaseProperty(object):
    def __init__(self, name):
        self.name = name

    def fget(self, obj):
        if hasattr(obj.optimizer.constant, self.name):
            return getattr(obj.optimizer.constant, self.name)
        elif hasattr(obj.optimizer.variable, self.name):
            return getattr(obj.optimizer.variable, self.name)

    def fset(self, obj, value):
        if is_prior(value):
            setattr(obj.optimizer.variable, self.name, value)
            try:
                delattr(obj.optimizer.constant, self.name)
            except AttributeError:
                pass
        else:
            setattr(obj.optimizer.constant, self.name, value)
            try:
                delattr(obj.optimizer.variable, self.name)
            except AttributeError:
                pass

    def fdel(self, obj):
        try:
            delattr(obj.optimizer.constant, self.name)
        except AttributeError:
            pass

        try:
            delattr(obj.optimizer.variable, self.name)
        except AttributeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.fget(obj)

    def __set__(self, obj, value):
        self.fset(obj, value)

    def __delete__(self, obj):
        return self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)


class ParallelPhase(Phase):
    parallel = PhaseProperty("parallel")

    def __init__(self, parallel=None, optimizer_class=nl.MultiNest, ci_datas_extractor=parallel_extractor, columns=None,
                 mask_function=default_mask_function, phase_name="parallel_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor, columns=columns,
                         rows=None, mask_function=mask_function, phase_name=phase_name)
        self.parallel = parallel

    class Analysis(Phase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

        def fit(self, instance):
            """
            Runs the analysis. Determine how well the supplied cti_params fits the image.

            Params
            ----------
            instance
                A model instance

            Returns
            -------
            result: Result
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            self.try_log(instance)
            cti_params = self.cti_params_for_instance(instance)
            fitter = fitting.CIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings)
            return fitter.likelihood

        def fit_pool(self, instance):
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
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):

            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)

            masks = list(map(lambda ci_data: ci_data.mask, self.ci_datas))

            # self.output_ci_regions_binned_across_serial(ci_post_ctis, masks, '/ci_post_cti_')
            # self.output_ci_regions_binned_across_serial(residuals, masks, '/residuals_')
            # self.output_ci_regions_binned_across_serial(chi_squareds, masks, '/chi_squareds_')
            #
            # self.output_parallel_trails_binned_across_serial(ci_post_ctis, masks, '/ci_post_cti_')
            # self.output_parallel_trails_binned_across_serial(residuals, masks, '/residuals_')
            # self.output_parallel_trails_binned_across_serial(chi_squareds, masks, '/chi_squareds_')

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

        def cti_params_for_instance(self, instance):
            return arctic_params.ArcticParams(parallel=instance.parallel)

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
    parallel = PhaseProperty("parallel")
    hyp_ci_regions = PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = PhaseProperty("hyp_parallel_trails")

    def __init__(self, parallel=None, hyp_ci_regions=None, hyp_parallel_trails=None,
                 optimizer_class=nl.MultiNest, ci_datas_extractor=parallel_extractor, columns=None,
                 mask_function=default_mask_function, phase_name="parallel_hyper_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(parallel=parallel, optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor,
                         columns=columns, mask_function=mask_function, phase_name=phase_name)
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_parallel_trails = hyp_parallel_trails
        self.has_noise_scalings = True

    class Analysis(ParallelPhase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

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
            self.try_log(instance)
            hyper_fitter = self.fitter_analysis_for_instance(instance)
            return hyper_fitter.scaled_likelihood

        def fit_pool(self, instance):
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
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_parallel_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            pass

            # fitter = self.fitter_analysis_for_instance(instance)
            # scaled_chi_squares = fitter.scaled_chi_squareds
            # masks = list(map(lambda ci_data: ci_data.mask, self.ci_datas))

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Parallel CTI::\n{}\n\n "
                "Hyper Parameters:\n{}\n{}\n".format(instance.parallel, instance.hyp_ci_regions,
                                                     instance.hyp_parallel_trails))

        def fitter_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas, cti_params=cti_params, cti_settings=self.cti_settings,
                                         hyper_noises=[instance.hyp_ci_regions,
                                                       instance.hyp_parallel_trails])

        def fitter_analysis_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas_analysis, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                                                       instance.hyp_parallel_trails])

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase

            Parameters
            ----------

            galaxy_images: [ndarray]
                A collection of images created by each individual galaxy which taken together form the full model image
            """
            super().__init__(constant, likelihood, variable, analysis)


class ParallelHyperOnlyPhase(ParallelHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def __init__(self, parallel=None, hyp_ci_regions=None, hyp_parallel_trails=None,
                 optimizer_class=nl.MultiNest, ci_datas_extractor=default_extractor, columns=None,
                 mask_function=default_mask_function, phase_name="parallel_hyper_only_phase"):
        super().__init__(parallel=parallel, hyp_ci_regions=hyp_ci_regions,
                         hyp_parallel_trails=hyp_parallel_trails, optimizer_class=optimizer_class,
                         ci_datas_extractor=ci_datas_extractor, columns=columns, mask_function=mask_function,
                         phase_name=phase_name)

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class ParallelHyper(ParallelHyperPhase):
            def pass_priors(self, previous_results):
                self.parallel = previous_results.last.constant.parallel

        phase = ParallelHyper(optimizer_class=nl.MultiNest, hyp_ci_regions=ci_hyper.HyperCINoise,
                              hyp_parallel_trails=ci_hyper.HyperCINoise,
                              ci_datas_extractor=self.ci_datas_extractor, phase_name=self.phase_name)

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        hyper_result = phase.run(ci_datas, cti_settings, previous_results, pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.likelihood, hyper_result.variable,
                                     analysis)


class SerialPhase(Phase):
    serial = PhaseProperty("serial")

    def __init__(self, serial=None, optimizer_class=nl.MultiNest, ci_datas_extractor=serial_extractor, columns=None,
                 rows=None, mask_function=default_mask_function, phase_name="serial_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor, columns=columns,
                         rows=rows, mask_function=mask_function, phase_name=phase_name)
        self.serial = serial

    class Analysis(Phase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

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
            self.try_log(instance)
            cti_params = self.cti_params_for_instance(instance)
            fitter = fitting.CIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings)
            return fitter.likelihood

        def fit_pool(self, instance):
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
            #    logger.debug("\nRunning analysis for... \n\ncti_model:\n{}\n\n".format( "\n\n".join(map(str, cti_params))))
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):

            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)

            masks = list(map(lambda ci_data: ci_data.mask, self.ci_datas))

            # self.output_ci_regions_binned_across_parallel(ci_post_ctis, masks, '/ci_post_cti_')
            # self.output_ci_regions_binned_across_parallel(residuals, masks, '/residuals_')
            # self.output_ci_regions_binned_across_parallel(chi_squareds, masks, '/chi_squareds_')
            #
            # self.output_serial_trails_binned_across_parallel(ci_post_ctis, masks, '/ci_post_cti_')
            # self.output_serial_trails_binned_across_parallel(residuals, masks, '/residuals_')
            # self.output_serial_trails_binned_across_parallel(chi_squareds, masks, '/chi_squareds_')

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

        def cti_params_for_instance(self, instance):
            return arctic_params.ArcticParams(serial=instance.serial)

        def noise_scalings_for_instance(self, instance):
            """

            First noises scaling images are of the charge injection regions.
            Second noises scaling images are of the non-charge injection regions in the serial calibration ci_frame"""
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
    serial = PhaseProperty("serial")
    hyp_ci_regions = PhaseProperty("hyp_ci_regions")
    hyp_serial_trails = PhaseProperty("hyp_serial_trails")

    def __init__(self, serial=None, hyp_ci_regions=None, hyp_serial_trails=None, optimizer_class=nl.MultiNest,
                 ci_datas_extractor=serial_extractor, columns=None, rows=None, mask_function=default_mask_function,
                 phase_name="serial_hyper_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(serial=serial, optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor,
                         columns=columns, rows=rows, mask_function=mask_function, phase_name=phase_name)
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_serial_trails = hyp_serial_trails
        self.has_noise_scalings = True

    class Analysis(SerialPhase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

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
            self.try_log(instance)
            hyper_fitter = self.hyper_fitter_analysis_for_instance(instance)
            return hyper_fitter.scaled_likelihood

        def fit_pool(self, instance):
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
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_serial_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            pass

            # fitter = self.fitter_analysis_for_instance(instance)
            # scaled_chi_squares = fitter.scaled_chi_squareds
            # masks = list(map(lambda ci_data: ci_data.mask, self.ci_datas))

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... \n\n"
                "Serial CTI::\n{}\n\n "
                "Hyper Parameters:\n{}\n{}\n".format(instance.serial, instance.hyp_ci_regions,
                                                     instance.hyp_serial_trails))

        def hyper_fitter_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                                                       instance.hyp_serial_trails])

        def hyper_fitter_analysis_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas_analysis, cti_params=cti_params,
                                         cti_settings=self.cti_settings, hyper_noises=[instance.hyp_ci_regions,
                                                                                       instance.hyp_serial_trails])

    class Result(SerialPhase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase

            Parameters
            ----------

            galaxy_images: [ndarray]
                A collection of images created by each individual galaxy which taken together form the full model image
            """
            super().__init__(constant, likelihood, variable, analysis)


class SerialHyperOnlyPhase(SerialHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def __init__(self, serial=None, hyp_ci_regions=None, hyp_serial_trails=None,
                 optimizer_class=nl.MultiNest, ci_datas_extractor=default_extractor, columns=None, rows=None,
                 mask_function=default_mask_function, phase_name="serial_hyper_only_phase"):
        super().__init__(serial=serial, hyp_ci_regions=hyp_ci_regions,
                         hyp_serial_trails=hyp_serial_trails, optimizer_class=optimizer_class,
                         ci_datas_extractor=ci_datas_extractor, columns=columns, rows=rows, mask_function=mask_function,
                         phase_name=phase_name)

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class SerialHyper(SerialHyperPhase):
            def pass_priors(self, previous_results):
                self.serial = previous_results.last.constant.serial

        phase = SerialHyper(optimizer_class=nl.MultiNest, hyp_ci_regions=ci_hyper.HyperCINoise,
                            hyp_serial_trails=ci_hyper.HyperCINoise,
                            ci_datas_extractor=self.ci_datas_extractor, phase_name=self.phase_name)

        phase.optimizer.n_live_points = 20
        phase.optimizer.sampling_efficiency = 0.8

        hyper_result = phase.run(ci_datas, cti_settings, previous_results, pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.likelihood, hyper_result.variable,
                                     analysis)


class ParallelSerialPhase(Phase):
    parallel = PhaseProperty("parallel")
    serial = PhaseProperty("serial")

    def __init__(self, parallel=None, serial=None, optimizer_class=nl.MultiNest,
                 ci_datas_extractor=parallel_serial_extractor, mask_function=default_mask_function,
                 phase_name="parallel_serial_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor, columns=None,
                         rows=None, mask_function=mask_function, phase_name=phase_name)
        self.parallel = parallel
        self.serial = serial

    class Analysis(Phase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

        def fit(self, instance):
            """
            Runs the analysis. Determine how well the supplied cti_params fits the image.

            Params
            ----------
            instance
                A model instance

            Returns
            -------
            result: Result
                An object comprising the final cti_params instances generated and a corresponding likelihood
            """
            self.try_log(instance)
            cti_params = self.cti_params_for_instance(instance)
            fitter = fitting.CIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings)
            return fitter.likelihood

        def fit_pool(self, instance):
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
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti, cti_params=cti_params, cti_settings=self.cti_settings)
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            fitter, ci_post_ctis, residuals, chi_squareds = super().visualize(instance, suffix, during_analysis)

            return fitter, ci_post_ctis, residuals, chi_squareds

        @classmethod
        def log(cls, instance):
            logger.debug(
                "\nRunning CTI analysis for... "
                "\n\nParallel CTI::\n{}"
                "\n\nSerial CTI::\n{}\n\n"
                    .format(instance.parallel, instance.serial))

        def cti_params_for_instance(self, instance):
            return arctic_params.ArcticParams(parallel=instance.parallel, serial=instance.serial)

        def noise_scalings_for_instance(self, instance):
            """

            First noises scaling images are of the charge injection regions.
            Second noises scaling images are of the non-charge injection regions in the parallel calibration ci_frame"""
            cti_params = self.cti_params_for_instance(instance)
            fitter = fitting.CIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings)
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
    parallel = PhaseProperty("parallel")
    serial = PhaseProperty("serial")
    hyp_ci_regions = PhaseProperty("hyp_ci_regions")
    hyp_parallel_trails = PhaseProperty("hyp_parallel_trails")
    hyp_serial_trails = PhaseProperty("hyp_serial_trails")
    hyp_parallel_serial_trails = PhaseProperty("hyp_parallel_serial_trails")

    def __init__(self, parallel=None, serial=None, hyp_ci_regions=None, hyp_parallel_trails=None,
                 hyp_serial_trails=None, hyp_parallel_serial_trails=None, optimizer_class=nl.MultiNest,
                 ci_datas_extractor=parallel_serial_extractor, mask_function=default_mask_function,
                 phase_name="parallel_serial_hyper_phase"):
        """
        A phase with a simple source/CTI model

        Parameters
        ----------
        CTI_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that acts as a gravitational CTI
        source_galaxy: g.Galaxy | gp.GalaxyPrior
            A galaxy that is being CTIed
        optimizer_class: class
            The class of a non-linear optimizer
        sub_grid_size: int
            The side length of the subgrid
        """
        super().__init__(parallel=parallel, serial=serial, optimizer_class=optimizer_class,
                         ci_datas_extractor=ci_datas_extractor, mask_function=mask_function, phase_name=phase_name)
        self.hyp_ci_regions = hyp_ci_regions
        self.hyp_parallel_trails = hyp_parallel_trails
        self.hyp_serial_trails = hyp_serial_trails
        self.hyp_parallel_serial_trails = hyp_parallel_serial_trails
        self.has_noise_scalings = True

    class Analysis(ParallelPhase.Analysis):

        def __init__(self, ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results=None, pool=None):
            """
            An analysis object. Once set up with the image ci_data (image, mask, noises) and pre-cti image it takes a set of \
            objects describing a model and determines how well they fit the image.

            Params
            ----------
            ci_data : [CIImage.CIImage]
                The charge injection ci_data-sets.
            """
            super().__init__(ci_datas, ci_datas_analysis, cti_settings, phase_name, previous_results, pool)

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
            self.try_log(instance)
            hyper_fitter = self.fitter_analysis_for_instance(instance)
            return hyper_fitter.scaled_likelihood

        def fit_pool(self, instance):
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
            cti_params = self.cti_params_for_instance(instance)
            pipe_cti_pass = partial(pipe_cti_hyper, cti_params=cti_params, cti_settings=self.cti_settings,
                                    hyper_noises=[instance.hyp_ci_regions, instance.hyp_parallel_trails,
                                                  instance.hyp_serial_trails,
                                                  instance.hyp_parallel_serial_trails])
            return np.sum(list(self.pool.map(pipe_cti_pass, self.ci_pipe_data)))

        def visualize(self, instance, suffix, during_analysis):
            pass

            # for i in range(len(self.ci_datas)):
            #
            #     self.output_array_as_fits(self.ci_datas[i].noise_scaling_map[0], "noise_scaling_" + str(i) + '_0')
            #     self.output_array_as_fits(self.ci_datas[i].noise_scaling_map[1], "noise_scaling_" + str(i) + '_1')
            #     self.output_array_as_fits(self.ci_datas[i].noise_scaling_map[2], "noise_scaling_" + str(i) + '_2')
            #     self.output_array_as_fits(self.ci_datas[i].noise_scaling_map[3], "noise_scaling_" + str(i) + '_3')

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

        def fitter_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas, cti_params=cti_params, cti_settings=self.cti_settings,
                                         hyper_noises=[instance.hyp_ci_regions, instance.hyp_parallel_trails,
                                                       instance.hyp_serial_trails,
                                                       instance.hyp_parallel_serial_trails])

        def fitter_analysis_for_instance(self, instance):
            cti_params = self.cti_params_for_instance(instance)
            return fitting.HyperCIFitter(self.ci_datas_analysis, cti_params=cti_params, cti_settings=self.cti_settings,
                                         hyper_noises=[instance.hyp_ci_regions,
                                                       instance.hyp_parallel_trails,
                                                       instance.hyp_serial_trails,
                                                       instance.hyp_parallel_serial_trails])

    class Result(Phase.Result):

        def __init__(self, constant, likelihood, variable, analysis):
            """
            The result of a phase

            Parameters
            ----------

            galaxy_images: [ndarray]
                A collection of images created by each individual galaxy which taken together form the full model image
            """
            super().__init__(constant, likelihood, variable, analysis)


class ParallelSerialHyperOnlyPhase(ParallelSerialHyperPhase, HyperOnly):
    """
    Fit only the CTI galaxy light.
    """

    def __init__(self, parallel=None, serial=None,
                 hyp_ci_regions=None, hyp_parallel_trails=None,
                 hyp_serial_trails=None, hyp_parallel_serial_trails=None,
                 optimizer_class=nl.MultiNest, ci_datas_extractor=parallel_serial_extractor,
                 mask_function=default_mask_function,
                 phase_name="parallel_serial_hyper_only_phase"):
        super().__init__(parallel=parallel, serial=serial,
                         hyp_ci_regions=hyp_ci_regions,
                         hyp_parallel_trails=hyp_parallel_trails,
                         hyp_serial_trails=hyp_serial_trails,
                         hyp_parallel_serial_trails=hyp_parallel_serial_trails,
                         optimizer_class=optimizer_class, ci_datas_extractor=ci_datas_extractor,
                         mask_function=mask_function, phase_name=phase_name)

    def run(self, ci_datas, cti_settings, previous_results=None, pool=None):
        class ParallelSerialHyper(ParallelSerialHyperPhase):
            def pass_priors(self, previous_results):
                self.serial = previous_results[-1].constant.serial
                self.parallel = previous_results[-1].constant.parallel

        phase = ParallelSerialHyper(optimizer_class=nl.MultiNest,
                                    hyp_ci_regions=ci_hyper.HyperCINoise,
                                    hyp_parallel_trails=ci_hyper.HyperCINoise,
                                    hyp_serial_trails=ci_hyper.HyperCINoise,
                                    hyp_parallel_serial_trails=ci_hyper.HyperCINoise,
                                    phase_name=self.phase_name)

        phase.optimizer.n_live_points = 50
        phase.optimizer.sampling_efficiency = 0.5

        hyper_result = phase.run(ci_datas, cti_settings, previous_results, pool)

        analysis = self.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results,
                                      pool=pool)

        return self.__class__.Result(hyper_result.constant, hyper_result.likelihood, hyper_result.variable,
                                     analysis)


def pipe_cti(ci_pipe_data, cti_params, cti_settings):
    ci_datas_analysis = ci_data.CIDataAnalysis(images=[ci_pipe_data[0]], masks=[ci_pipe_data[1]],
                                               noises=[ci_pipe_data[2]],
                                               ci_pre_ctis=[ci_pipe_data[3]])
    fitter = fitting.CIFitter(ci_datas_analysis, cti_params=cti_params, cti_settings=cti_settings)
    return fitter.likelihood


def pipe_cti_hyper(ci_pipe_data, cti_params, cti_settings, hyper_noises):
    ci_datas_analysis = ci_data.CIDataAnalysis(images=[ci_pipe_data[0]], masks=[ci_pipe_data[1]],
                                               noises=[ci_pipe_data[2]],
                                               ci_pre_ctis=[ci_pipe_data[3]], noise_scalings=[ci_pipe_data[4]])
    fitter = fitting.HyperCIFitter(ci_datas_analysis, cti_params=cti_params, cti_settings=cti_settings,
                                   hyper_noises=hyper_noises)
    return fitter.scaled_likelihood
