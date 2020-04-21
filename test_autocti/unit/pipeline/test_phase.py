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


@pytest.fixture(name="cti_settings")
def make_cti_settings():
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
    def test__param_names(self):

        phase = ac.PhaseCI(phase_name="test_phase", non_linear_class=NLO)

        phase.parallel_traps = [af.PriorModel(ac.Trap), af.PriorModel(ac.Trap)]

        assert phase.optimizer.model.param_names == [
            "parallel_traps_0_trap_density",
            "parallel_traps_0_trap_lifetime",
            "parallel_traps_1_trap_density",
            "parallel_traps_1_trap_lifetime",
        ]

        phase = ac.PhaseCI(
            phase_name="test_phase",
            parallel_traps=[af.PriorModel(ac.Trap), af.PriorModel(ac.Trap)],
            non_linear_class=NLO,
        )

        assert phase.optimizer.model.param_names == [
            "parallel_traps_0_trap_density",
            "parallel_traps_0_trap_lifetime",
            "parallel_traps_1_trap_density",
            "parallel_traps_1_trap_lifetime",
        ]

    def test__make_analysis(self, phase, ci_data, cti_settings):
        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)
        assert analysis.last_results is None
        assert (analysis.masked_ci_dataset_extracted[0].image == ci_data.image).all()
        assert (
            analysis.masked_ci_dataset_extracted[0].noise_map == ci_data.noise_map
        ).all()
        assert (analysis.masked_ci_dataset_full[0].image == ci_data.image).all()
        assert (analysis.masked_ci_dataset_full[0].noise_map == ci_data.noise_map).all()
        assert analysis.cti_settings == cti_settings

    def test__make_analysis__default_mask_all_empty__cosmic_ray_map_masks_mask(
        self, phase, ci_data, cti_settings
    ):

        phase.cosmic_ray_parallel_buffer = 0
        phase.cosmic_ray_serial_buffer = 0
        phase.cosmic_ray_diagonal_buffer = 0

        ci_data.cosmic_ray_map = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[False, False, False], [False, True, False], [False, False, False]]
            )
        ).all()

        phase.cosmic_ray_parallel_buffer = 1
        phase.cosmic_ray_serial_buffer = 1
        phase.cosmic_ray_diagonal_buffer = 1

        ci_data.cosmic_ray_map = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[False, False, False], [False, True, True], [False, True, True]]
            )
        ).all()

        phase.cosmic_ray_parallel_buffer = 2
        phase.cosmic_ray_serial_buffer = 1
        phase.cosmic_ray_diagonal_buffer = 1

        ci_data.cosmic_ray_map = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[True, True, False], [True, True, False], [True, False, False]]
            )
        ).all()

    def test__make_analysis__uses_parallel_x1__serial_x1_front_edge_or_trials_for_mask(
        self, phase, ci_data, cti_settings
    ):

        phase.parallel_front_edge_mask_rows = (0, 1)
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[True, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = (0, 2)
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[False, False, False], [True, False, False], [True, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = (0, 1)
        phase.serial_trails_mask_columns = None

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[True, False, False], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = None
        phase.parallel_trails_mask_rows = None
        phase.serial_front_edge_mask_columns = None
        phase.serial_trails_mask_columns = (0, 2)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[False, True, True], [False, False, False], [False, False, False]]
            )
        ).all()

        phase.parallel_front_edge_mask_rows = (0, 1)
        phase.parallel_trails_mask_rows = (1, 2)
        phase.serial_front_edge_mask_columns = (0, 1)
        phase.serial_trails_mask_columns = (0, 2)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        assert (
            analysis.masked_ci_dataset_full[0].mask
            == np.array(
                [[True, True, True], [False, False, False], [True, False, False]]
            )
        ).all()

    def test__parallel_total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase.parallel_total_density_range = (1.0, 2.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=0.75, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.75, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=0.1, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=1.5, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=1.5, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (10.0, 15.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=12.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=2.0, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=9.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.9, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=14.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=2.0, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(parallel_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__serial_total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase.serial_total_density_range = (1.0, 2.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=0.75, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.75, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=0.1, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=1.5, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=1.5, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.serial_total_density_range = (10.0, 15.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=12.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=2.0, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=9.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.9, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        species_0 = ac.Trap(trap_density=14.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=2.0, trap_lifetime=2.0)
        cti_params = ac.ArcticParams(serial_traps=[species_0, species_1])

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__parallel_serial__total_density_within_values__if_not_true_raises_exception(
        self, phase, ci_data, cti_settings
    ):

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=0.75, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.75, trap_lifetime=2.0)
        species_2 = ac.Trap(trap_density=5.5, trap_lifetime=1.0)
        species_3 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)

        cti_params = ac.ArcticParams(
            parallel_traps=[species_0, species_1], serial_traps=[species_2, species_3]
        )

        analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=0.1, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)
        species_2 = ac.Trap(trap_density=5.5, trap_lifetime=1.0)
        species_3 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)

        cti_params = ac.ArcticParams(
            parallel_traps=[species_0, species_1], serial_traps=[species_2, species_3]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

        phase.parallel_total_density_range = (1.0, 2.0)
        phase.serial_total_density_range = (5.0, 6.0)

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)

        species_0 = ac.Trap(trap_density=1.0, trap_lifetime=1.0)
        species_1 = ac.Trap(trap_density=0.5, trap_lifetime=2.0)
        species_2 = ac.Trap(trap_density=6.0, trap_lifetime=1.0)
        species_3 = ac.Trap(trap_density=0.1, trap_lifetime=2.0)

        cti_params = ac.ArcticParams(
            parallel_traps=[species_0, species_1], serial_traps=[species_2, species_3]
        )

        with pytest.raises(exc.PriorException):
            analysis.check_total_density_within_range(cti_params=cti_params)

    def test__customize_constant(self, results, ci_data, cti_settings):
        class MyPhaseCI(ac.PhaseCI):
            def customize_priors(self, previous_results):
                self.parallel_traps = previous_results.last.instance.parallel_traps

        parallel = ac.Trap()

        setattr(results.instance, "parallel_traps", [parallel])

        results_collection = af.ResultsCollection()
        results_collection.add("first_phase", results)

        phase = MyPhaseCI(
            phase_name="test_phase", non_linear_class=NLO, parallel_traps=[parallel]
        )
        phase.make_analysis([ci_data], cti_settings, results=results_collection)

        assert phase.parallel_traps == [parallel]

    # noinspection PyUnresolvedReferences
    def test__data_extractors(self, ci_data):

        phase = ac.PhaseCI(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
        )

        ci_datas_masked = phase.masked_ci_dataset_extracted_from_ci_data(
            ci_data, ac.Mask.unmasked(shape_2d=ci_data.image.shape_2d)
        )

        assert isinstance(ci_datas_masked, ac.MaskedCIImaging)
        assert (ci_data.image == ci_datas_masked_image).all()
        assert (ci_data.noise_map == ci_datas_masked.noise_map).all()
        assert (ci_data.ci_pre_cti == ci_datas_masked.ci_pre_cti).all()

        phase = ac.PhaseCI(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            columns=1,
            non_linear_class=NLO,
        )

        ci_datas_masked = phase.masked_ci_dataset_extracted_from_ci_data(
            ci_data, ac.Mask.unmasked(shape_2d=ci_data.image.shape_2d)
        )

        assert isinstance(ci_datas_masked, ac.MaskedCIImaging)
        assert (ci_data.image[:, 0] == ci_datas_masked_image).all()
        assert (ci_data.noise_map[:, 0] == ci_datas_masked.noise_map).all()
        assert (ci_data.ci_pre_cti[:, 0] == ci_datas_masked.ci_pre_cti).all()

        phase = ac.PhaseCI(
            phase_name="test_phase",
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
        )

        ci_datas_masked = phase.masked_ci_dataset_extracted_from_ci_data(
            ci_data, ac.Mask.unmasked(shape_2d=ci_data.image.shape_2d)
        )

        assert isinstance(ci_datas_masked, ac.MaskedCIImaging)
        assert (ci_data.image == ci_datas_masked_image).all()
        assert (ci_data.noise_map == ci_datas_masked.noise_map).all()
        assert (ci_data.ci_pre_cti == ci_datas_masked.ci_pre_cti).all()

        phase = ac.PhaseCI(
            phase_name="test_phase",
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            rows=(0, 1),
            non_linear_class=NLO,
        )

        ci_datas_masked = phase.masked_ci_dataset_extracted_from_ci_data(
            ci_data, ac.Mask.unmasked(shape_2d=ci_data.image.shape_2d)
        )

        assert isinstance(ci_datas_masked, ac.MaskedCIImaging)
        assert (ci_data.image[0, :] == ci_datas_masked_image).all()
        assert (ci_data.noise_map[0, :] == ci_datas_masked.noise_map).all()
        assert (ci_data.ci_pre_cti[0, :] == ci_datas_masked.ci_pre_cti).all()

        phase = ac.PhaseCI(
            phase_name="test_phase",
            parallel_traps=[ac.Trap],
            parallel_ccd_volume=ac.CCDVolume,
            serial_traps=[ac.Trap],
            serial_ccd_volume=ac.CCDVolume,
            columns=1,
            rows=(0, 1),
            non_linear_class=NLO,
        )

        ci_datas_masked = phase.masked_ci_dataset_extracted_from_ci_data(
            ci_data, ac.Mask.unmasked(shape_2d=ci_data.image.shape_2d)
        )

        assert isinstance(ci_datas_masked, ac.MaskedCIImaging)
        assert (ci_data.image == ci_datas_masked_image).all()
        assert (ci_data.noise_map == ci_datas_masked.noise_map).all()
        assert (ci_data.ci_pre_cti == ci_datas_masked.ci_pre_cti).all()

    def test__cti_params_for_instance(self):

        instance = af.ModelInstance()
        instance.parallel_traps = [ac.Trap(trap_density=0.0)]
        cti_params = ac.cti_params_for_instance(instance)

        assert cti_params.parallel_traps == instance.parallel_traps

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

    def test__make_analysis__ci_regions_and_parallel_trail_scalars__noise_scaling_maps_list_are_setup_correctly(
        self, ci_data, cti_settings
    ):

        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
        )

        class Results(object):
            def __init__(self, last):
                self.last = last

        results = Results(last=result)

        phase2 = ac.PhaseCI(
            phase_name="test_phase_2",
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            hyper_noise_scalar_of_ci_regions=ac.CIHyperNoiseScalar,
            hyper_noise_scalar_of_parallel_trails=ac.CIHyperNoiseScalar,
            non_linear_class=NLO,
        )

        analysis = phase2.make_analysis(
            datasets=[ci_data, ci_data, ci_data, ci_data],
            cti_settings=cti_settings,
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

    def test__make_analysis__ci_region_and_serial_trail_scalars___noise_scaling_maps_list_are_setup_correctly(
        self, ci_data
    ):

        serial_settings = ac.Settings(well_depth=0, niter=1, express=1, n_levels=2000)
        cti_settings = ac.ArcticSettings(neomode="NEO", serial=serial_settings)

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
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
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
            cti_settings=cti_settings,
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
        cti_settings = ac.ArcticSettings(
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
            ci_datas=[ci_data, ci_data, ci_data, ci_data], cti_settings=cti_settings
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
            cti_settings=cti_settings,
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

    def test__extended_with_hyper_noise_phase(self, phase):

        phase_extended = phase.extend_with_hyper_noise_phases()
        assert type(phase_extended.hyper_phases[0]) == ac.HyperNoisePhase


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


class TestResult(object):
    # def test__fits_for_instance__uses_ci_data_fit(self, ci_data, cti_settings):
    #     parallel_traps = ac.Trap(trap_density=0.1, trap_lifetime=1.0)
    #     parallel_ccd_volume = ac.CCDVolume(well_notch_depth=0.1, well_fill_alpha=0.5, well_fill_beta=0.5,
    #                                      well_fill_gamma=0.5)
    #
    #     phase = ac.ParallelPhase(parallel_traps=[parallel_traps], parallel_ccd_volume=parallel_ccd_volume, columns=1,
    #                              phase_name='test_phase')
    #
    #     analysis = phase.make_analysis(ci_datas=[ci_data], cti_settings=cti_settings)
    #     instance = phase.model.instance_from_unit_vector([])
    #
    #     fits = analysis.fits_of_ci_data_extracted_for_instance(instance=instance)
    #     assert fits[0].ci_pre_cti.shape == (3, 1)
    #
    #     full_fits = analysis.fits_of_ci_data_full_for_instance(instance=instance)
    #     assert full_fits[0].ci_pre_cti.shape == (3, 3)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, ci_data, cti_settings
    ):

        parallel_traps = ac.Trap(trap_density=0.1, trap_lifetime=1.0)
        parallel_ccd_volume = ac.CCDVolume(
            well_notch_depth=0.1,
            well_fill_alpha=0.5,
            well_fill_beta=0.5,
            well_fill_gamma=0.5,
        )

        phase = ac.PhaseCI(
            parallel_traps=[parallel_traps],
            parallel_ccd_volume=parallel_ccd_volume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        analysis = phase.make_analysis(datasets=[ci_data], cti_settings=cti_settings)
        instance = phase.model.instance_from_unit_vector([])
        cti_params = ac.cti_params_for_instance(instance=instance)
        fit_figure_of_merit = analysis.fit(instance=instance)

        mask = phase.mask_function(shape=ci_data.image.shape)
        ci_datas_masked = [
            phase.masked_ci_dataset_extracted_from_ci_data(ci_data=d, mask=mask)
            for d, mask in zip([ci_data], [mask])
        ]
        fit = ac.CIFitImaging(
            masked_ci_imaging=ci_datas_masked[0],
            cti_params=cti_params,
            cti_settings=cti_settings,
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__results_of_phase_are_available_as_properties(self, ci_data, cti_settings):
        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert hasattr(result, "most_likely_extracted_fits")
        assert hasattr(result, "most_likely_full_fits")
        assert hasattr(result, "cti_settings")
        assert hasattr(result, "noise_scaling_maps_list_of_ci_regions")
        assert hasattr(result, "noise_scaling_maps_list_of_parallel_trails")
        assert hasattr(result, "noise_scaling_maps_list_of_serial_trails")
        assert hasattr(result, "noise_scaling_maps_list_of_serial_overscan_no_trails")

    def test__cti_settings_passed_as_result_correctly(self, ci_data, cti_settings):

        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert result.cti_settings == cti_settings

    def test__parallel_phase__noise_scaling_maps_list_of_result__are_correct(
        self, ci_data, cti_settings
    ):

        phase = ac.PhaseCI(
            parallel_traps=[af.PriorModel(ac.Trap)],
            parallel_ccd_volume=ac.CCDVolume,
            non_linear_class=NLO,
            phase_name="test_phase",
        )

        # The ci_region is [0, 1, 0, 1], therefore by changing the image at 0,0 to 2.0 there will be a residual of 1.0,
        # which for a noise_map entry of 2.0 gives a chi squared of 0.25..

        ci_data.image[0, 0] = 2.0
        ci_data.noise_map[0, 0] = 2.0

        result = phase.run(ci_datas=[ci_data], cti_settings=cti_settings)

        assert (
            result.noise_scaling_maps_list_of_ci_regions[0]
            == np.array([[0.25, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ).all()

        assert (
            result.noise_scaling_maps_list_of_parallel_trails[0] == np.zeros((3, 3))
        ).all()
        assert (
            result.noise_scaling_maps_list_of_serial_trails[0] == np.zeros((3, 3))
        ).all()
        assert (
            result.noise_scaling_maps_list_of_serial_overscan_no_trails[0]
            == np.zeros((3, 3))
        ).all()
