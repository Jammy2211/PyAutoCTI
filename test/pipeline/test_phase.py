from os import path

import numpy as np
import pytest
from autofit import conf
from autofit.mapper import model_mapper as mm
from autofit.mapper import prior_model
from autofit.optimize import non_linear as nl
from autofit.tools import phase_property
from autofit.tools.phase import ResultsCollection

from autocti import exc
from autocti.charge_injection import ci_data as data
from autocti.charge_injection import ci_fit
from autocti.charge_injection import ci_frame
from autocti.data import mask as msk
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
        self.regions = [ci_frame.Region(region=[0, 1, 0, 1])]


@pytest.fixture(name="phase")
def make_phase():
    return ph.ParallelPhase(optimizer_class=NLO)


@pytest.fixture(name="cti_settings")
def make_cti_settings():
    parallel_settings = arctic_settings.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
    return arctic_settings.ArcticSettings(neomode='NEO', parallel=parallel_settings)


@pytest.fixture(name="frame_geometry")
def make_frame_geometry():
    return ci_frame.QuadGeometryEuclid.bottom_left()


@pytest.fixture(name="ci_pattern")
def make_ci_pattern():
    return MockPattern()


@pytest.fixture(name="ci_data")
def make_ci_data(frame_geometry, ci_pattern):
    image = np.ones((3, 3))
    noise = np.ones((3, 3))
    ci_pre_cti = np.ones((3, 3))
    return data.CIData(image=image, noise_map=noise, ci_pre_cti=ci_pre_cti, ci_pattern=ci_pattern,
                       ci_frame=frame_geometry)


@pytest.fixture(name="results")
def make_results():
    return MockResults(ci_post_ctis=[np.ones((10, 10)), np.ones((10, 10))])


@pytest.fixture(name="results_collection")
def make_results_collection(results):
    return ResultsCollection([results])


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

    def test__make_analysis(self, phase, ci_data, cti_settings):
        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        assert analysis.last_results is None
        assert (analysis.ci_datas_extracted[0].image == ci_data.image).all()
        assert (analysis.ci_datas_extracted[0].noise_map == ci_data.noise_map).all()
        assert (analysis.ci_datas_full[0].image == ci_data.image).all()
        assert (analysis.ci_datas_full[0].noise_map == ci_data.noise_map).all()
        assert analysis.cti_settings == cti_settings

    def test__parallel_phase__if_trap_lifetime_not_ascending__raises_exception(self, phase, ci_data, cti_settings):
        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1, species_2])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=5.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__serial_phase__if_trap_lifetime_not_ascending__raises_exception(self, ci_data, cti_settings):
        phase = ph.SerialPhase(optimizer_class=NLO)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1, species_2])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=5.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1, species_2])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__parallel_serial_phase__if_trap_lifetime_not_ascending__raises_exception(self, ci_data, cti_settings):
        phase = ph.ParallelSerialPhase(optimizer_class=NLO)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species_0, parallel_species_1],
                                                serial_species=[serial_species_0, serial_species_1])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        parallel_species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[parallel_species_0, parallel_species_1, parallel_species_2],
            serial_species=[serial_species_0, serial_species_1, serial_species_2])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species_0, parallel_species_1],
                                                serial_species=[serial_species_0, serial_species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species_0, parallel_species_1],
                                                serial_species=[serial_species_0, serial_species_1])
        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__customize_constant(self, results, ci_data, cti_settings):
        class MyPhase(ph.ParallelPhase):
            def pass_priors(self, previous_results):
                self.parallel_species = previous_results.last.constant.parallel_species

        parallel = arctic_params.Species()

        setattr(results.constant, "parallel_species", [parallel])

        phase = MyPhase(optimizer_class=NLO, parallel_species=[parallel])
        phase.make_analysis([ci_data], cti_settings, previous_results=ResultsCollection([results]))

        assert phase.parallel_species == [parallel]

    # noinspection PyUnresolvedReferences
    def test__default_data_extractor(self, ci_data, phase):
        ci_datas_fit = phase.extract_ci_data(ci_data, msk.Mask.empty_for_shape(ci_data.image.shape))

        assert isinstance(ci_datas_fit, data.MaskedCIData)
        assert (ci_data.image == ci_datas_fit.image).all()
        assert (ci_data.noise_map == ci_datas_fit.noise_map).all()
        assert (ci_data.ci_pre_cti == ci_datas_fit.ci_pre_cti).all()

    def test__phase_property(self):
        class MyPhase(ph.ParallelPhase):
            prop = phase_property.PhaseProperty("prop")

        parallel = arctic_params.Species()

        phase = MyPhase(optimizer_class=NLO, parallel_species=[parallel])

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


class TestHyperPhase(object):

    def test__make_analysis(self, phase, ci_data, cti_settings):
        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        assert analysis.last_results is None
        assert (analysis.ci_datas_extracted[0].image == ci_data.image).all()
        assert (analysis.ci_datas_extracted[0].noise_map == ci_data.noise_map).all()
        assert (analysis.ci_datas_full[0].image == ci_data.image).all()
        assert (analysis.ci_datas_full[0].noise_map == ci_data.noise_map).all()
        assert analysis.cti_settings == cti_settings

    def test_hyper_phase(self):
        class MockResult:
            noise_scaling_maps_of_ci_regions = 1
            noise_scaling_maps_of_parallel_trails = 2
            noise_scaling_maps_of_serial_trails = 3
            noise_scaling_maps_of_serial_overscan_above_trails = 4

        noise_scaling_maps = ph.ParallelHyperPhase().noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [1, 2]

        noise_scaling_maps = ph.SerialHyperPhase().noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [1, 3]

        noise_scaling_maps = ph.ParallelSerialHyperPhase().noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [1, 2, 3, 4]


class TestResult(object):

    def test_results(self):
        results = ResultsCollection([1, 2, 3])
        assert results == [1, 2, 3]
        assert results.last == 3
        assert results.first == 1

    def test__fits_for_instance__uses_ci_data_fit(self, ci_data, cti_settings):
        parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        parallel_ccd = arctic_params.CCD(well_notch_depth=0.1, well_fill_alpha=0.5, well_fill_beta=0.5,
                                         well_fill_gamma=0.5)

        phase = ph.ParallelPhase(parallel_species=[parallel_species], parallel_ccd=parallel_ccd, columns=1,
                                 phase_name='test_phase')

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        instance = phase.constant

        fits = analysis.fits_of_ci_data_extracted_for_instance(instance=instance)
        assert fits[0].ci_pre_cti.shape == (3, 1)

        full_fits = analysis.fits_of_ci_data_full_for_instance(instance=instance)
        assert full_fits[0].ci_pre_cti.shape == (3, 3)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(self, ci_data, cti_settings):
        parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        parallel_ccd = arctic_params.CCD(well_notch_depth=0.1, well_fill_alpha=0.5, well_fill_beta=0.5,
                                         well_fill_gamma=0.5)

        phase = ph.ParallelPhase(parallel_species=[parallel_species], parallel_ccd=parallel_ccd,
                                 phase_name='test_phase')

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        instance = phase.constant
        cti_params = ph.cti_params_for_instance(instance=instance)
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(shape=ci_data.image.shape)
        ci_datas_fit = [phase.extract_ci_data(data=d, mask=mask) for d, mask in zip([ci_data], [mask])]
        fit = ci_fit.CIFit(masked_ci_data=ci_datas_fit[0], cti_params=cti_params, cti_settings=cti_settings)

        assert fit.likelihood == fit_figure_of_merit

    def test__results_of_phase_are_available_as_properties(self, ci_data, cti_settings):
        phase = ph.ParallelPhase(optimizer_class=NLO,
                                 parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                                 parallel_ccd=arctic_params.CCD, phase_name='test_phase')

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert hasattr(result, 'most_likely_extracted_fits')
        assert hasattr(result, 'most_likely_full_fits')
        assert hasattr(result, 'noise_scaling_maps_of_ci_regions')
        assert hasattr(result, 'noise_scaling_maps_of_parallel_trails')
        assert hasattr(result, 'noise_scaling_maps_of_serial_trails')
        assert hasattr(result, 'noise_scaling_maps_of_serial_overscan_above_trails')

    def test__parallel_phase__noise_scaling_maps_of_images_are_correct(self, ci_data, cti_settings):
        phase = ph.ParallelPhase(optimizer_class=NLO,
                                 parallel_species=[prior_model.PriorModel(arctic_params.Species)],
                                 parallel_ccd=arctic_params.CCD, phase_name='test_phase')

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (result.noise_scaling_maps_of_ci_regions[0] == np.array([[0.25, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.0],
                                                                        [0.0, 0.0, 0.0]])).all()

        assert (result.noise_scaling_maps_of_parallel_trails[0] == np.zeros((3, 3))).all()
        assert (result.noise_scaling_maps_of_serial_trails[0] == np.zeros((3, 3))).all()
        assert (result.noise_scaling_maps_of_serial_overscan_above_trails[0] == np.zeros((3, 3))).all()
