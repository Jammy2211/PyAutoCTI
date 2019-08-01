from os import path

import numpy as np
import pytest

import autofit as af
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
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    af.conf.instance = af.conf.Config(
        "{}/../test_files/configs/phase".format(directory)
    )


class MockResults(object):
    def __init__(self, ci_post_ctis):
        self.ci_post_ctis = ci_post_ctis
        self.variable = af.ModelMapper()
        self.constant = af.ModelMapper()


class NLO(af.NonLinearOptimizer):
    def fit(self, analysis):
        class Fitness(object):
            def __init__(self, instance_from_physical_vector):
                self.result = None
                self.instance_from_physical_vector = instance_from_physical_vector

            def __call__(self, vector):
                instance = self.instance_from_physical_vector(vector)

                likelihood = analysis.fit(instance)
                self.result = af.Result(instance, likelihood)

                # Return Chi squared
                return -2 * likelihood

        fitness_function = Fitness(self.variable.instance_from_physical_vector)
        fitness_function(self.variable.prior_count * [0.5])

        return fitness_function.result


class MockPattern(object):
    def __init__(self):
        self.regions = [ci_frame.Region(region=[0, 1, 0, 1])]


@pytest.fixture(name="phase")
def make_phase():
    return ph.ParallelPhase(phase_name="test_phase", optimizer_class=NLO)


@pytest.fixture(name="cti_settings")
def make_cti_settings():
    parallel_settings = arctic_settings.Settings(
        well_depth=0, niter=1, express=1, n_levels=2000
    )
    return arctic_settings.ArcticSettings(neomode="NEO", parallel=parallel_settings)


@pytest.fixture(name="frame_geometry")
def make_frame_geometry():
    return ci_frame.FrameGeometry.euclid_bottom_left()


@pytest.fixture(name="ci_pattern")
def make_ci_pattern():
    return MockPattern()


@pytest.fixture(name="ci_data")
def make_ci_data(frame_geometry, ci_pattern):
    image = np.ones((3, 3))
    noise = np.ones((3, 3))
    ci_pre_cti = np.ones((3, 3))
    frame = ci_frame.ChInj(frame_geometry=frame_geometry, ci_pattern=ci_pattern)
    return data.CIData(
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
    def test_param_names(self, phase):

        phase.parallel_species = [
            af.PriorModel(arctic_params.Species),
            af.PriorModel(arctic_params.Species),
        ]

        assert phase.optimizer.variable.param_names == [
            "parallel_species_0_trap_density",
            "parallel_species_0_trap_lifetime",
            "parallel_species_1_trap_density",
            "parallel_species_1_trap_lifetime",
        ]

        phase = ph.ParallelPhase(
            phase_name="test_phase",
            parallel_species=[
                af.PriorModel(arctic_params.Species),
                af.PriorModel(arctic_params.Species),
            ],
            optimizer_class=NLO,
        )

        assert phase.optimizer.variable.param_names == [
            "parallel_species_0_trap_density",
            "parallel_species_0_trap_lifetime",
            "parallel_species_1_trap_density",
            "parallel_species_1_trap_lifetime",
        ]

    def test__make_analysis(self, phase, ci_data, cti_settings):
        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        assert analysis.last_results is None
        assert (analysis.ci_datas_extracted[0].image == ci_data.image).all()
        assert (analysis.ci_datas_extracted[0].noise_map == ci_data.noise_map).all()
        assert (analysis.ci_datas_full[0].image == ci_data.image).all()
        assert (analysis.ci_datas_full[0].noise_map == ci_data.noise_map).all()
        assert analysis.cti_settings == cti_settings

    def test__make_analysis__uses_mask_function(self, phase, ci_data, cti_settings):
        def mask_function(shape, ci_frame):
            return np.full(shape=shape, fill_value=False)

        phase.mask_function = mask_function

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask == np.full(shape=(3, 3), fill_value=False)
        ).all()

        def mask_function(shape, ci_frame):
            return np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )

        phase.mask_function = mask_function

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

    def test__make_analysis__default_mask_all_empty__cosmic_ray_image_masks_mask(
        self, phase, ci_data, cti_settings
    ):

        phase.cosmic_ray_parallel_buffer = 0
        phase.cosmic_ray_serial_buffer = 0
        phase.cosmic_ray_diagonal_buffer = 0

        ci_data.cosmic_ray_image = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        phase.cosmic_ray_parallel_buffer = 1
        phase.cosmic_ray_serial_buffer = 1
        phase.cosmic_ray_diagonal_buffer = 1

        ci_data.cosmic_ray_image = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[False, False, False], [False, True, True], [False, True, True]]
            )
        ).all()

        phase.cosmic_ray_parallel_buffer = 2
        phase.cosmic_ray_serial_buffer = 1
        phase.cosmic_ray_diagonal_buffer = 1

        ci_data.cosmic_ray_image = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[True, True, False], [True, True, False], [True, False, False]]
            )
        ).all()

    def test__make_analysis__uses_parallel_and_serial_front_edge_or_trials_for_mask(
        self, phase, ci_data, cti_settings
    ):

        phase.parallel_front_edge_mask_rows = (0, 1)
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[True, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = (0, 2)
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[False, False, False], [True, False, False], [True, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = (0, 1)
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[True, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = (0, 2)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = (0, 1)
        phase.parallel_trails_mask_rows = (1, 2)
        phase.serial_front_edge_mask_columns = (0, 1)
        phase.serial_trails_mask_columns = (0, 2)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.ci_datas_full[0].mask
            == np.array(
                [[True, True, True], [False, False, False], [True, False, False]]
            )
        ).all()

    def test__parallel_phase__if_trap_lifetime_not_ascending__raises_exception(
        self, phase, ci_data, cti_settings
    ):
        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1, species_2]
        )

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=5.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__serial_phase__if_trap_lifetime_not_ascending__raises_exception(
        self, ci_data, cti_settings
    ):
        phase = ph.SerialPhase(phase_name="test_phase", optimizer_class=NLO)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            serial_species=[species_0, species_1, species_2]
        )

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(
            serial_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            serial_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=5.0)
        species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=4.0)
        species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            serial_species=[species_0, species_1, species_2]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__parallel_serial_phase__if_trap_lifetime_not_ascending__raises_exception(
        self, ci_data, cti_settings
    ):
        phase = ph.ParallelSerialPhase(phase_name="test_phase", optimizer_class=NLO)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[parallel_species_0, parallel_species_1],
            serial_species=[serial_species_0, serial_species_1],
        )

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        parallel_species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_2 = arctic_params.Species(trap_density=0.0, trap_lifetime=3.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[
                parallel_species_0,
                parallel_species_1,
                parallel_species_2,
            ],
            serial_species=[serial_species_0, serial_species_1, serial_species_2],
        )

        analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[parallel_species_0, parallel_species_1],
            serial_species=[serial_species_0, serial_species_1],
        )

        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

        parallel_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_0 = arctic_params.Species(trap_density=0.0, trap_lifetime=2.0)
        serial_species_1 = arctic_params.Species(trap_density=0.0, trap_lifetime=1.0)
        cti_params = arctic_params.ArcticParams(
            parallel_species=[parallel_species_0, parallel_species_1],
            serial_species=[serial_species_0, serial_species_1],
        )
        with pytest.raises(exc.PriorException):
            analysis.check_trap_lifetimes_are_ascending(cti_params=cti_params)

    def test__parallel_phase__total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase.parallel_total_density_range = (1.0, 2.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.75, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.75, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=1.5, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (10.0, 15.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=12.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=2.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=9.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.9, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=14.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=2.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(parallel_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__serial_phase__total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase = ph.SerialPhase(phase_name="test_phase", optimizer_class=NLO)

        phase.serial_total_density_range = (1.0, 2.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.75, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.75, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=1.5, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=1.5, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.serial_total_density_range = (10.0, 15.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=12.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=2.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=9.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.9, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = arctic_params.Species(trap_density=14.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=2.0, trap_lifetime=2.0)
        cti_params = arctic_params.ArcticParams(serial_species=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__parallel_serial_phase__total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase = ph.ParallelSerialPhase(phase_name="test_phase", optimizer_class=NLO)

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.75, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.75, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=5.5, trap_lifetime=1.0)
        species_3 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)

        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1],
            serial_species=[species_2, species_3],
        )

        analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=5.5, trap_lifetime=1.0)
        species_3 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)

        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1],
            serial_species=[species_2, species_3],
        )

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)

        species_0 = arctic_params.Species(trap_density=1.0, trap_lifetime=1.0)
        species_1 = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)
        species_2 = arctic_params.Species(trap_density=6.0, trap_lifetime=1.0)
        species_3 = arctic_params.Species(trap_density=0.1, trap_lifetime=2.0)

        cti_params = arctic_params.ArcticParams(
            parallel_species=[species_0, species_1],
            serial_species=[species_2, species_3],
        )

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__customize_constant(self, results, ci_data, cti_settings):
        class MyPhase(ph.ParallelPhase):
            def pass_priors(self, previous_results):
                self.parallel_species = previous_results.last.constant.parallel_species

        parallel = arctic_params.Species()

        setattr(results.constant, "parallel_species", [parallel])

        results_collection = af.ResultsCollection()
        results_collection.add("first_phase", results)

        phase = MyPhase(
            phase_name="test_phase", optimizer_class=NLO, parallel_species=[parallel]
        )
        phase.make_analysis([ci_data], cti_settings, results=results_collection)

        assert phase.parallel_species == [parallel]

    # noinspection PyUnresolvedReferences
    def test__default_data_extractor(self, ci_data, phase):
        ci_datas_fit = phase.extract_ci_data(
            ci_data, msk.Mask.empty_for_shape(ci_data.image.shape)
        )

        assert isinstance(ci_datas_fit, data.MaskedCIData)
        assert (ci_data.image == ci_datas_fit.image).all()
        assert (ci_data.noise_map == ci_datas_fit.noise_map).all()
        assert (ci_data.ci_pre_cti == ci_datas_fit.ci_pre_cti).all()

    def test__duplication(self):
        parallel = arctic_params.Species()

        phase = ph.ParallelPhase(phase_name="test_phase", parallel_species=[parallel])

        ph.ParallelPhase(phase_name="test_phase", parallel_species=[])

        assert phase.parallel_species is not None

    def test__cti_params_for_instance(self):

        instance = af.ModelInstance()
        instance.parallel_species = [arctic_params.Species(trap_density=0.0)]
        cti_params = ph.cti_params_for_instance(instance)

        assert cti_params.parallel_species == instance.parallel_species


class MockResult:

    noise_scaling_maps_of_ci_regions = [1]
    noise_scaling_maps_of_parallel_trails = [2]
    noise_scaling_maps_of_serial_trails = [3]
    noise_scaling_maps_of_serial_overscan_above_trails = [4]


class MockInstance:
    parallel_species = 1
    parallel_ccd = 2
    serial_species = 3
    serial_ccd = 4
    hyper_noise_scalars = [5]


@pytest.fixture(name="parallel_hyper_analysis")
def make_parallel_hyper_analysis():
    phase = ph.ParallelHyperPhase(phase_name="test_phase")
    return phase.make_analysis([], [], results=[MockResult])


@pytest.fixture(name="serial_hyper_analysis")
def make_serial_hyper_analysis():
    phase = ph.SerialHyperPhase(phase_name="test_phase")
    return phase.make_analysis([], [], results=[MockResult])


@pytest.fixture(name="parallel_serial_hyper_analysis")
def make_parallel_serial_hyper_analysis():
    phase = ph.ParallelSerialHyperPhase(phase_name="test_phase")
    return phase.make_analysis([], [], results=[MockResult])


class TestHyperPhase(object):
    def test_types(self):
        parallel_phase = ph.ParallelHyperPhase(phase_name="test_phase")
        serial_phase = ph.SerialHyperPhase(phase_name="test_phase")
        parallel_serial_phase = ph.ParallelSerialHyperPhase(phase_name="test_phase")

        assert isinstance(parallel_phase, ph.ParallelPhase)
        assert isinstance(parallel_phase, ph.HyperPhase)

        assert isinstance(serial_phase, ph.SerialPhase)
        assert isinstance(serial_phase, ph.HyperPhase)

        assert isinstance(parallel_serial_phase, ph.ParallelSerialPhase)
        assert isinstance(parallel_serial_phase, ph.HyperPhase)

    def test_hyper_phase_make_analysis(
        self,
        parallel_hyper_analysis,
        serial_hyper_analysis,
        parallel_serial_hyper_analysis,
    ):
        assert isinstance(parallel_hyper_analysis, ph.HyperPhase.Analysis)
        assert isinstance(serial_hyper_analysis, ph.HyperPhase.Analysis)
        assert isinstance(parallel_serial_hyper_analysis, ph.HyperPhase.Analysis)

    def test_describe(
        self,
        parallel_hyper_analysis,
        serial_hyper_analysis,
        parallel_serial_hyper_analysis,
    ):

        assert """
Running CTI analysis for... 

Parallel CTI: 
Parallel Species:
1

Parallel CCD
2

Hyper Parameters:
5

""" == parallel_hyper_analysis.describe(
            MockInstance
        )

        assert """
Running CTI analysis for... 

Serial CTI: 
Serial Species:
3

Serial CCD
4

Hyper Parameters:
5

""" == serial_hyper_analysis.describe(
            MockInstance
        )

        assert """
Running CTI analysis for... 

Parallel CTI: 
Parallel Species:
1

Parallel CCD
2

Serial CTI: 
Serial Species:
3

Serial CCD
4

Hyper Parameters:
5

""" == parallel_serial_hyper_analysis.describe(
            MockInstance
        )

    def test__make_analysis(self, phase, ci_data, cti_settings):

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        assert analysis.last_results is None
        assert (analysis.ci_datas_extracted[0].image == ci_data.image).all()
        assert (analysis.ci_datas_extracted[0].noise_map == ci_data.noise_map).all()
        assert (analysis.ci_datas_full[0].image == ci_data.image).all()
        assert (analysis.ci_datas_full[0].noise_map == ci_data.noise_map).all()
        assert analysis.cti_settings == cti_settings

    def test_noise_scaling_map_extraction(self):

        noise_scaling_maps = ph.ParallelHyperPhase(
            phase_name="test_phase"
        ).noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [[1, 2]]

        noise_scaling_maps = ph.SerialHyperPhase(
            phase_name="test_phase"
        ).noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [[1, 3]]

        noise_scaling_maps = ph.ParallelSerialHyperPhase(
            phase_name="test_phase"
        ).noise_scaling_maps_from_result(MockResult)
        assert noise_scaling_maps == [[1, 2, 3, 4]]

    def test_hyper_phase(self):
        phase = ph.ParallelHyperPhase(phase_name="test_phase")
        assert len(phase.hyper_noise_scalars) == 2
        assert len(phase.variable.priors) == 2

        instance = phase.variable.instance_from_unit_vector([0.5, 0.5])

        assert list(instance.hyper_noise_scalars) == [5.0, 5.0]


class TestResult(object):
    # def test__fits_for_instance__uses_ci_data_fit(self, ci_data, cti_settings):
    #     parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
    #     parallel_ccd = arctic_params.CCD(well_notch_depth=0.1, well_fill_alpha=0.5, well_fill_beta=0.5,
    #                                      well_fill_gamma=0.5)
    #
    #     phase = ph.ParallelPhase(parallel_species=[parallel_species], parallel_ccd=parallel_ccd, columns=1,
    #                              phase_name='test_phase')
    #
    #     analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
    #     instance = phase.variable.instance_from_unit_vector([])
    #
    #     fits = analysis.fits_of_ci_data_extracted_for_instance(instance=instance)
    #     assert fits[0].ci_pre_cti.shape == (3, 1)
    #
    #     full_fits = analysis.fits_of_ci_data_full_for_instance(instance=instance)
    #     assert full_fits[0].ci_pre_cti.shape == (3, 3)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, ci_data, cti_settings
    ):

        parallel_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        parallel_ccd = arctic_params.CCD(
            well_notch_depth=0.1,
            well_fill_alpha=0.5,
            well_fill_beta=0.5,
            well_fill_gamma=0.5,
        )

        phase = ph.ParallelPhase(
            parallel_species=[parallel_species],
            parallel_ccd=parallel_ccd,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
        instance = phase.variable.instance_from_unit_vector([])
        cti_params = ph.cti_params_for_instance(instance=instance)
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(shape=ci_data.image.shape)
        ci_datas_fit = [
            phase.extract_ci_data(data=d, mask=mask)
            for d, mask in zip([ci_data], [mask])
        ]
        fit = ci_fit.CIFit(
            masked_ci_data=ci_datas_fit[0],
            cti_params=cti_params,
            cti_settings=cti_settings,
        )

        assert fit.likelihood == fit_figure_of_merit

    def test__results_of_phase_are_available_as_properties(self, ci_data, cti_settings):
        phase = ph.ParallelPhase(
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert hasattr(result, "most_likely_extracted_fits")
        assert hasattr(result, "most_likely_full_fits")
        assert hasattr(result, "noise_scaling_maps_of_ci_regions")
        assert hasattr(result, "noise_scaling_maps_of_parallel_trails")
        assert hasattr(result, "noise_scaling_maps_of_serial_trails")
        assert hasattr(result, "noise_scaling_maps_of_serial_overscan_above_trails")

    def test__parallel_phase__noise_scaling_maps_of_images_are_correct(
        self, ci_data, cti_settings
    ):

        phase = ph.ParallelPhase(
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            result.noise_scaling_maps_of_ci_regions[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            result.noise_scaling_maps_of_parallel_trails[0] == np.zeros((3, 3))
        ).all()
        assert (result.noise_scaling_maps_of_serial_trails[0] == np.zeros((3, 3))).all()
        assert (
            result.noise_scaling_maps_of_serial_overscan_above_trails[0]
            == np.zeros((3, 3))
        ).all()

    def test__parallel_hyper_phase__noise_scaling_maps_are_setup_correctly(
        self, ci_data, cti_settings
    ):

        phase = ph.ParallelPhase(
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
        )

        phase2 = ph.ParallelHyperPhase(
            phase_name="test_phase_2",
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            optimizer_class=NLO,
        )

        noise_scaling_maps = phase2.noise_scaling_maps_from_result(result=result)

        assert (
            noise_scaling_maps[0][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[1][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[2][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[3][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (noise_scaling_maps[0][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[1][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[2][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[3][1] == np.zeros((3, 3))).all()

    def test__serial_hyper_phase__noise_scaling_maps_are_setup_correctly(self, ci_data):

        serial_settings = arctic_settings.Settings(
            well_depth=0, niter=1, express=1, n_levels=2000
        )
        cti_settings = arctic_settings.ArcticSettings(
            neomode="NEO", serial=serial_settings
        )

        phase = ph.SerialPhase(
            serial_species=[af.PriorModel(arctic_params.Species)],
            serial_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
        )

        phase2 = ph.SerialHyperPhase(
            serial_species=[af.PriorModel(arctic_params.Species)],
            serial_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase_2",
        )

        noise_scaling_maps = phase2.noise_scaling_maps_from_result(result=result)

        assert (
            noise_scaling_maps[0][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[1][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[2][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[3][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (noise_scaling_maps[0][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[1][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[2][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[3][1] == np.zeros((3, 3))).all()

    def test__parallel_and_serial_hyper_phase__noise_scaling_maps_are_setup_correctly(
        self, ci_data
    ):

        parallel_settings = arctic_settings.Settings(
            well_depth=0, niter=1, express=1, n_levels=2000
        )
        serial_settings = arctic_settings.Settings(
            well_depth=0, niter=1, express=1, n_levels=2000
        )
        cti_settings = arctic_settings.ArcticSettings(
            neomode="NEO", parallel=parallel_settings, serial=serial_settings
        )

        phase = ph.ParallelSerialPhase(
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            serial_species=[af.PriorModel(arctic_params.Species)],
            serial_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
        )

        phase2 = ph.ParallelSerialHyperPhase(
            parallel_species=[af.PriorModel(arctic_params.Species)],
            parallel_ccd=arctic_params.CCD,
            serial_species=[af.PriorModel(arctic_params.Species)],
            serial_ccd=arctic_params.CCD,
            optimizer_class=NLO,
            phase_name="test_phase_2",
        )

        noise_scaling_maps = phase2.noise_scaling_maps_from_result(result=result)

        assert (
            noise_scaling_maps[0][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[1][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[2][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            noise_scaling_maps[3][0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (noise_scaling_maps[0][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[1][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[2][1] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[3][1] == np.zeros((3, 3))).all()

        assert (noise_scaling_maps[0][2] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[1][2] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[2][2] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[3][2] == np.zeros((3, 3))).all()

        assert (noise_scaling_maps[0][3] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[1][3] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[2][3] == np.zeros((3, 3))).all()
        assert (noise_scaling_maps[3][3] == np.zeros((3, 3))).all()
