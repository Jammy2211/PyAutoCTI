from autofit.core import non_linear as nl
from autofit.core import model_mapper as mm
from autocti.pipeline import phase as ph
from autocti.data.charge_injection import ci_frame
from autocti.data.charge_injection import ci_data
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autofit import conf

import numpy as np
import pytest
from os import path

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result "
    "either in an error or a different result.")

directory = path.dirname(path.realpath(__file__))

general_conf = '{}/../test_files/configs/phase'.format(directory)
conf.instance.general = conf.NamedConfig("{}/general.ini".format(general_conf))

class MockResults(object):
    def __init__(self, ci_post_ctis):
        self.ci_post_ctis = ci_post_ctis
        self.constant = mm.ModelInstance()
        self.variable = mm.ModelMapper()


class NLO(nl.NonLinearOptimizer):

    def fit(self, analysis):

        class Fitness(object):

            def __init__(self, instance_from_physical_vector, constant):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector
                self.constant = constant

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)
                for key, value in self.constant.__dict__.items():
                    setattr(instance, key, value)

                likelihood = analysis.fit(instance)
                self.result = nl.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector, self.constant)
        fitness_function(self.variable.prior_count * [0.5])

        return fitness_function.result


class MockPattern(object):

    def __init__(self):

        self.regions = [MockRegion()]


class MockRegion(object):

    def __init__(self):
        self.x0 = 0
        self.x1 = 1
        self.y0 = 0
        self.y1 = 1

    def ci_regions_frame_from_frame(self):
        pass

    def non_ci_regions_frame_from_frame(self):
        pass

    def add_region_from_image_to_array(self, image, array):
        pass

    def set_region_on_array_to_zeros(self, array):
        pass


@pytest.fixture(name="phase")
def make_phase():
    return ph.ParallelPhase(optimizer_class=NLO, parallel=arctic_params.ParallelOneSpecies,
                            ci_datas_extractor=ph.default_extractor)

@pytest.fixture(name="cti_settings")
def make_cti_settings():
    parallel_settings = arctic_settings.ParallelSettings(well_depth=0, niter=1, express=1, n_levels=2000)
    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)

@pytest.fixture(name="ci_geometry")
def make_ci_geometry():
    return ci_frame.CIQuadGeometryEuclidBL()

@pytest.fixture(name="ci_pattern")
def make_ci_pattern():
    return MockPattern()

@pytest.fixture(name="ci_datas")
def make_ci_datas(ci_geometry, ci_pattern):
    images = [ci_data.CIImage(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=np.ones((3, 3)))]
    masks = [ci_data.CIMask.empty_for_shape(shape=(3, 3), frame_geometry=ci_geometry, ci_pattern=ci_pattern)]
    noises = [np.ones((3,3))]
    ci_pre_ctis = [ci_data.CIPreCTI(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=np.ones((3, 3)))]
    return ci_data.CIData(images, masks, noises, ci_pre_ctis)

@pytest.fixture(name="results")
def make_results():
    return MockResults(ci_post_ctis=[np.ones((10, 10)), np.ones((10, 10))])

@pytest.fixture(name="results_collection")
def make_results_collection(results):
    return ph.ResultsCollection([results])


class TestPhase(object):

    def test__set_constants(self, phase):
        parallel = arctic_params.ParallelOneSpecies()
        phase.parallel = parallel
        assert phase.optimizer.constant.parallel == parallel
        assert not hasattr(phase.optimizer.variable, "parallel")

    def test__set_variables(self, phase):
        parallel = arctic_params.ParallelOneSpecies
        phase.parallel = parallel
        assert phase.optimizer.variable.parallel == phase.parallel
        assert not hasattr(phase.optimizer.constant, "parallel")

    def test__mask_analysis(self, phase, ci_datas, cti_settings):

        analysis = phase.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings)
        assert analysis.last_results is None
        assert analysis.ci_datas == ci_datas
        assert analysis.cti_settings == cti_settings

    # def test__fit(self, phase, ci_datas, cti_settings):
    #
    #     phase.parallel = arctic_params.ParallelOneSpecies()
    #     result = phase.run(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=None)
    #     assert isinstance(result.constant.parallel, arctic_params.ParallelOneSpecies)

    def test__customize_constant(self, results, ci_datas, cti_settings):

        class MyPhase(ph.ParallelPhase):
            def pass_priors(self, previous_results):
                self.parallel = previous_results.last.constant.parallel

        parallel = arctic_params.ParallelOneSpecies()

        setattr(results.constant, "parallel", parallel)

        phase = MyPhase(optimizer_class=NLO, parallel=parallel, ci_datas_extractor=ph.default_extractor)
        phase.make_analysis(ci_datas, cti_settings, previous_results=ph.ResultsCollection([results]))

        assert phase.parallel == parallel

    def test__customize_variable(self, results, ci_datas, cti_settings):

        class MyPhase(ph.ParallelPhase):
            def pass_priors(self, previous_results):
                self.parallel = previous_results.last.variable.parallel

        parallel_prior = arctic_params.ParallelOneSpecies

        setattr(results.variable, "parallel", parallel_prior)

        phase = MyPhase(optimizer_class=NLO, parallel=results, ci_datas_extractor=ph.default_extractor)
        phase.make_analysis(ci_datas, cti_settings, previous_results=ph.ResultsCollection([results]))

        assert phase.parallel == results.variable.parallel

    def test__default_data_extractor(self, ci_datas, phase):

        ci_data_analysis = phase.ci_datas_extractor(ci_datas, mask_function=ph.default_mask_function)

        assert type(ci_data_analysis) == ci_data.CIDataAnalysis
        assert (ci_datas[0].image == ci_data_analysis[0].image).all()
        assert (ci_datas[0].mask == ci_data_analysis[0].mask).all()
        assert (ci_datas[0].noise == ci_data_analysis[0].noise).all()
        assert (ci_datas[0].ci_pre_cti == ci_data_analysis[0].ci_pre_cti).all()

    def test__phase_property(self):

        class MyPhase(ph.ParallelPhase):
            prop = ph.PhaseProperty("prop")

        parallel = arctic_params.ParallelOneSpecies()

        phase = MyPhase(optimizer_class=NLO, parallel=parallel, ci_datas_extractor=ph.default_extractor)

        phase.prop = arctic_params.ParallelOneSpecies

        assert phase.variable.prop == phase.prop

        phase.prop = parallel

        assert phase.constant.prop == parallel
        assert not hasattr(phase.variable, "prop")

        phase.prop = arctic_params.ParallelOneSpecies
        assert not hasattr(phase.constant, "prop")

    def test__duplication(self):

        parallel = arctic_params.ParallelOneSpecies()

        phase = ph.ParallelPhase(parallel=parallel)

        ph.ParallelPhase(parallel=None)

        assert phase.parallel is not None

    def test__cti_params_for_instance(self, ci_datas, cti_settings):

        phase = ph.ParallelPhase(ci_datas_extractor=ph.default_extractor)
        analysis = phase.make_analysis(ci_datas, cti_settings)
        instance = mm.ModelInstance()
        instance.parallel = arctic_params.ParallelOneSpecies(trap_densities=(0.0,))
        cti_params = analysis.cti_params_for_instance(instance)

        assert cti_params.parallel == instance.parallel

#
# class TestParallelPhase(object):
#
#     def test__analysis__image_and_model_identical__likelihood_is_noise_term(self, cti_settings, ci_geometry):
#
#         ci_pattern = pattern.CIPatternUniform(normalization=1000.0, regions=[(0, 1, 0, 2)])
#         ci_images = [ci_data.CIImage(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=10.0 * np.ones((2, 2)))]
#         ci_noises = [ci_frame.ci_frame.from_single_value(value=2.0, shape=(2, 2), frame_geometry=ci_geometry,
#                                                        ci_pattern=ci_pattern)]
#         ci_masks = [ci_data.CIMask.empty_for_shape(frame_geometry=ci_geometry, shape=(2, 2), ci_pattern=ci_pattern)]
#         ci_pre_ctis = [ci_data.CIPreCTI(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=10.0 * np.ones((2, 2)))]
#         ci_datas = ci_data.CIData(images=ci_images, masks=ci_masks, noises=ci_noises, ci_pre_ctis=ci_pre_ctis)
#
#         phase = ph.ParallelPhase(parallel=arctic_params.ParallelOneSpecies, optimizer_class=NLO)
#
#         analysis = ph.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings)
#
#         likelihood = analysis.fit(parallel=arctic_params.ParallelOneSpecies(trap_densities=(0.0,)))
#
#         chi_sq_term = 0
#         noise_term = 4.0 * np.log(2 * np.pi * 4.0)
#
#         assert likelihood == -0.5 * (chi_sq_term + noise_term)
#
#     def test__image_and_pre_cti_not_identical__likelihood_is_chi_sq_plus_noise_term(self, ci_geometry, cti_settings):
#
#         phase = ph.ParallelPhase(parallel=arctic_params.ParallelOneSpecies, optimizer_class=NLO)
#
#         ci_pattern = pattern.CIPatternUniform(normalization=1000.0, regions=[(0, 1, 0, 2)])
#
#         ci_images = [ci_data.CIImage(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=9.0*np.ones((2,2)))]
#         ci_masks = [ci_data.CIMask.empty_for_shape(frame_geometry=ci_geometry, shape=(2, 2), ci_pattern=ci_pattern)]
#         ci_noises = [ci_frame.ci_frame.from_single_value(value=2.0, shape=(2, 2), frame_geometry=ci_geometry,
#                                                ci_pattern=ci_pattern)]
#         ci_pre_ctis = [ci_data.CIPreCTI(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=10.0 * np.ones((2, 2)))]
#         ci_datas = ci_data.CIData(images=ci_images, masks=ci_masks, noises=ci_noises, ci_pre_ctis=ci_pre_ctis)
#
#         analysis = ph.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings)
#
#         likelihood = analysis.fit(parallel=arctic_params.ParallelOneSpecies(trap_densities=(0.0,)))
#
#         chi_sq_term = 4.0 * ((1.0 / 2.0) ** 2.0)
#         noise_term = 4.0 * np.log(2 * np.pi * 4.0)
#
#         assert likelihood == pytest.approx(-0.5 * (chi_sq_term + noise_term), 1e-4)


# class TestParallelHyperPhase(object):
#
#     def test__analysis_scales_noise(self, ci_geometry, cti_settings):
#
#         ci_pattern = pattern.CIPatternUniform(normalization=1000.0, regions=[(0, 1, 0, 2)])
#         ci_images = [ci_data.CIImage(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=10.0 * np.ones((2, 2)))]
#         ci_masks = [ci_data.CIMask.empty_for_shape(frame_geometry=ci_geometry, shape=(2, 2), ci_pattern=ci_pattern)]
#         ci_noises = [ci_frame.ci_frame.from_single_value(value=2.0, shape=(2, 2), frame_geometry=ci_geometry,
#                                                        ci_pattern=ci_pattern)]
#         ci_pre_ctis = [ci_data.CIPreCTI(frame_geometry=ci_geometry, ci_pattern=ci_pattern, array=10.0 * np.ones((2, 2)))]
#         ci_datas = ci_data.CIData(images=ci_images, masks=ci_masks, noises=ci_noises, ci_pre_ctis=ci_pre_ctis)
#
#         noise_scalings = [[np.array([[1.0, 2.0], [3.0, 4.0]])]]
#         previous_results.noise_scalings = noise_scalings
#
#         phase = ph.ParallelHyperPhase(parallel=arctic_params.ParallelOneSpecies,
#                                          hyper_noise_ci_regions=CIHyper.HyperCINoise,
#                                          hyper_noise_parallel_non_ci_regions=CIHyper.HyperCINoise,
#                                          optimizer_class=NLO)
#
#         analysis = ph.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings, previous_results=previous_results)
#
#         likelihood = analysis.fit(parallel=arctic_params.ParallelOneSpecies(trap_densities=(0.0,)),
#                                   hyper_noise_ci_regions=CIHyper.HyperCINoise(scale_factor=1.0),
#                                   hyper_noise_parallel_non_ci_regions=CIHyper.HyperCINoise(scale_factor=0.0))
#
#         chi_sq_term = 0
#         noise_term = np.log(2 * np.pi * (2.0 + 1.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0) + \
#                      np.log(2 * np.pi * (2.0 + 3.0) ** 2.0) + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)
#
#         assert likelihood == -0.5 * (chi_sq_term + noise_term)

class TestResult(object):

    def test_results(self):

        results = ph.ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1