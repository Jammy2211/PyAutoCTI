from os import path

import numpy as np
import pytest
from autofit import conf
from autofit.core import model_mapper as mm
from autofit.core import non_linear as nl
from autofit.core import phase_property

from autocti.data import mask as msk
from autocti.data.charge_injection import ci_data
from autocti.data.charge_injection import ci_frame
from autocti.model import arctic_params
from autocti.model import arctic_settings
from autocti.pipeline import phase as ph

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
    return ph.ParallelPhase(optimizer_class=NLO, ci_datas_extractor=ph.default_extractor)


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
    masks = [msk.Mask.empty_for_shape(shape=(3, 3), frame_geometry=ci_geometry, ci_pattern=ci_pattern)]
    noises = [np.ones((3, 3))]
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
        parallel_species = arctic_params.Species()
        phase.parallel_species = [parallel_species]
        assert phase.optimizer.constant.parallel_species == [parallel_species]
        assert phase.optimizer.variable.parallel_species != [parallel_species]

    def test__set_variables(self, phase):
        parallel = [mm.PriorModel(arctic_params.Species)]
        phase.parallel_species = parallel
        assert phase.optimizer.variable.parallel_species == phase.parallel_species
        assert phase.optimizer.constant.parallel_species != [parallel]

    def test__mask_analysis(self, phase, ci_datas, cti_settings):
        analysis = phase.make_analysis(ci_datas=ci_datas, cti_settings=cti_settings)
        assert analysis.last_results is None
        assert analysis.ci_datas == ci_datas
        assert analysis.cti_settings == cti_settings

    def test__customize_constant(self, results, ci_datas, cti_settings):
        class MyPhase(ph.ParallelPhase):
            def pass_priors(self, previous_results):
                self.parallel_species = previous_results.last.constant.parallel_species

        parallel = arctic_params.Species()

        setattr(results.constant, "parallel_species", [parallel])

        phase = MyPhase(optimizer_class=NLO, parallel_species=[parallel], ci_datas_extractor=ph.default_extractor)
        phase.make_analysis(ci_datas, cti_settings, previous_results=ph.ResultsCollection([results]))

        assert phase.parallel_species == [parallel]

    # noinspection PyUnresolvedReferences
    def test__default_data_extractor(self, ci_datas, phase):
        ci_data_analysis = phase.ci_datas_extractor(ci_datas, mask_function=ph.default_mask_function)

        assert type(ci_data_analysis) == ci_data.CIDataAnalysis
        assert (ci_datas[0].image == ci_data_analysis[0].image).all()
        assert (ci_datas[0].mask == ci_data_analysis[0].mask).all()
        assert (ci_datas[0].noise == ci_data_analysis[0].noise).all()
        assert (ci_datas[0].ci_pre_cti == ci_data_analysis[0].ci_pre_cti).all()

    def test__phase_property(self):
        class MyPhase(ph.ParallelPhase):
            prop = phase_property.PhaseProperty("prop")

        parallel = arctic_params.Species()

        phase = MyPhase(optimizer_class=NLO, parallel_species=[parallel], ci_datas_extractor=ph.default_extractor)

        phase.prop = arctic_params.Species

        assert phase.variable.prop == phase.prop

        phase.prop = [parallel]

        assert phase.constant.prop == [parallel]
        assert not hasattr(phase.variable, "prop")

        phase.prop = arctic_params.Species
        assert not hasattr(phase.constant, "prop")

    def test__duplication(self):
        parallel = arctic_params.Species()

        phase = ph.ParallelPhase(parallel_species=[parallel])

        ph.ParallelPhase(parallel_species=[])

        assert phase.parallel_species is not None

    def test__cti_params_for_instance(self):
        instance = mm.ModelInstance()
        instance.parallel_species = [arctic_params.Species(trap_density=0.0)]
        cti_params = ph.cti_params_for_instance(instance)

        assert cti_params.parallel_species == instance.parallel_species


class TestResult(object):

    def test_results(self):
        results = ph.ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1
