from os import path

import numpy as np
import pytest

import autofit as af
import autocti as ac
from autocti.util import exc

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config("{}/files/configs/phase".format(directory))


class MockResults(object):
    def __init__(self, ci_post_ctis):
        self.ci_post_ctis = ci_post_ctis
        self.model = af.ModelMapper()
        self.instance = af.ModelMapper()


class NLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_vector):
                self.result = None
                self.instance_from_vector = instance_from_vector

            def __call__(self, vector):
                instance = self.instance_from_vector(vector)

                log_likelihood = analysis.fit(instance)
                self.result = af.Result(instance, log_likelihood)

                # Return Chi squared
                return -2 * log_likelihood

        fitness_function = Fitness(self.model.instance_from_vector)
        fitness_function(self.model.prior_count * [0.5])

        return fitness_function.result


class MockPattern(object):
    def __init__(self):
        self.regions = [ac.Region(region=[0, 1, 0, 1])]


@pytest.fixture(name="phase")
def make_phase():
    return ac.PhaseCI(
        phase_name="test_phase",
        parallel_traps=ac.Trap,
        parallel_ccd_volume=ac.CCDVolume,
        non_linear_class=NLO,
    )


@pytest.fixture(name="clocker")
def make_clocker():
    parallel_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
    return ac.ArcticSettings(neomode="NEO", parallel=parallel_settings)


@pytest.fixture(name="frame_geometry")
def make_frame_geometry():
    return ac.FrameGeometry.bottom_left()


@pytest.fixture(name="ci_pattern")
def make_ci_pattern():
    return MockPattern()


@pytest.fixture(name="ci_data")
def make_ci_data(frame_geometry, ci_pattern):
    image = np.ones((3, 3))
    noise = np.ones((3, 3))
    ci_pre_cti = np.ones((3, 3))
    frame = ac.CIFrame(frame_geometry=frame_geometry, ci_pattern=ci_pattern)
    return ac.CIImaging(
        image=image,
        noise_map=noise,
        ci_pre_cti=ci_pre_cti,
        ci_pattern=ci_pattern,
        ci_frame=frame,
    )


@pytest.fixture(name="results")
def make_results():
    return MockResults(ci_post_ctis=[np.ones((10, 10)), np.ones((10, 10))])


@pytest.fixture(name="results_collection")
def make_results_collection():
    return af.ResultsCollection()


class TestPhase(object):
    def test__hyper_noise_scalar_properties_of_phase(self):
        phase = ac.PhaseCI(
            phase_name="test_phase",
            hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ac.CIHyperNoiseScalar,
        )
        assert len(phase.hyper_noise_scalars) == 2
        assert len(phase.model.priors) == 2

        instance = phase.model.instance_from_unit_vector([0.5, 0.5])

        assert instance.hyper_noise_scalar_of_ci_regions == 5.0
        assert instance.hyper_noise_scalar_of_parallel_trails == 5.0

    def test__make_analysis__ci_region_and_serial_trail_scalars___noise_scaling_maps_list_are_setup_correctly(
        self, ci_data
    ):

        serial_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
        clocker = ac.ArcticSettings(neomode="NEO", serial=serial_settings)

        phase = ac.PhaseCI(
            serial_traps=[af.PriorModel(ac.Trap)],
            serial_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], clocker=clocker
        )

        class Results(object):
            def __init__(self, last):
                self.last = last

        results = Results(last=result)

        phase2 = ac.PhaseCI(
            serial_traps=[af.PriorModel(ac.Trap)],
            serial_ccd_volume=ac.CCDVolume,
            hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_trails=ac.CIHyperNoiseScalar,
            non_linear_class=NLO,
            phase_name="test_phase_2",
        )

        analysis = phase2.make_analysis(
            datasets=[ci_data, ci_data, ci_data, ci_data],
            clocker=clocker,
            results=results,
        )

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()

    def test__make_analysis__all_4_scalars__noise_scaling_maps_list_are_setup_correctly(
        self, ci_data
    ):

        parallel_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
        serial_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
        clocker = ac.ArcticSettings(
            neomode="NEO", parallel=parallel_settings, serial=serial_settings
        )

        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[af.PriorModel(ac.Trap)],
            serial_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], clocker=clocker
        )

        class Results(object):
            def __init__(self, last):
                self.last = last

        results = Results(last=result)

        phase2 = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[af.PriorModel(ac.Trap)],
            serial_ccd_volume=ac.CCDVolume,
            hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_trails=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_serial_overscan_no_trails=ac.CIHyperNoiseScalar,
            non_linear_class=NLO,
            phase_name="test_phase_2",
        )

        analysis = phase2.make_analysis(
            datasets=[ci_data, ci_data, ci_data, ci_data],
            clocker=clocker,
            results=results,
        )

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[1] == np.zeros((3, 3))
        ).all()

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[2] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[2] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[2] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[2] == np.zeros((3, 3))
        ).all()

        assert (
            analysis.masked_ci_dataset_full[0].noise_scaling_maps[3] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[1].noise_scaling_maps[3] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[2].noise_scaling_maps[3] == np.zeros((3, 3))
        ).all()
        assert (
            analysis.masked_ci_dataset_full[3].noise_scaling_maps[3] == np.zeros((3, 3))
        ).all()


class MockResult:

    noise_scaling_maps_list_of_ci_regions = [1]
    noise_scaling_maps_list_of_parallel_trails = [2]
    noise_scaling_maps_list_of_serial_trails = [3]
    noise_scaling_maps_list_of_serial_overscan_no_trails = [4]


class MockInstance:
    parallel_traps = 1
    parallel_ccd_volume = 2
    serial_traps = 3
    serial_ccd_volume = 4
    hyper_noise_scalars = [5]
