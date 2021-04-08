# import pytest
# from astropy import cosmology as cosmo
#
# import autofit as af
# import autocti as ac
# from autocti.pipeline.phase import extensions
# from autocti.charge_injection import ci_hyper
#
#
# @pytest.fixture(name="lens_galaxy")
# def simulate_lens_galaxy():
#     return ac.Galaxy(
#         redshift=1.0, light=ac.lp.SphSersic(), mass=ac.mp.SphIsothermal()
#     )
#
#
# @pytest.fixture(name="source_galaxy")
# def make_source_galaxy():
#     return ac.Galaxy(redshift=2.0, light=ac.lp.SphSersic())
#
#
# @pytest.fixture(name="all_galaxies")
# def make_all_galaxies(lens_galaxy, source_galaxy):
#     galaxies = af.ModelInstance()
#     galaxies.lens = lens_galaxy
#     galaxies.source = source_galaxy
#     return galaxies
#
#
# @pytest.fixture(name="instance")
# def make_instance(all_galaxies):
#     instance = af.ModelInstance()
#     instance.galaxies = all_galaxies
#     return instance
#
#
# @pytest.fixture(name="result")
# def make_result(masked_imaging_7x7, instance):
#     return phase_imaging.PhaseImaging.Result(
#         constant=instance,
#         figure_of_merit=1.0,
#         previous_model=af.ModelMapper(),
#         gaussian_tuples=None,
#         analysis=phase_imaging.PhaseImaging.Analysis(
#             masked_imaging=masked_imaging_7x7,
#             cosmology=cosmo.Planck15,
#             positions_threshold=1.0,
#         ),
#         search=None,
#     )
#
#
# class MostLikelyFit(object):
#     def __init__(self, model_image_2d):
#         self.model_image_2d = model_image_2d
#
#
# class MockResult(object):
#     def __init__(self, max_log_likelihood_fit=None):
#         self.max_log_likelihood_fit = max_log_likelihood_fit
#         self.analysis = MockAnalysis()
#         self.model = af.ModelMapper()
#         self.clocker = None
#
#
# class MockAnalysis(object):
#     pass
#
#
# # noinspection PyAbstractClass
# class MockOptimizer(af.NonLinearSearch):
#     def __init__(
#         self,
#         name="mock_search",
#         phase_tag="tag",
#         folders=tuple(),
#         model_mapper=None,
#     ):
#         super().__init__(
#             folders=folders,
#             phase_tag=phase_tag,
#             name=name,
#             model_mapper=model_mapper,
#         )
#
#     def fit(self, analysis):
#         # noinspection PyTypeChecker
#         return af.Result(None, analysis.fit(None), None)
#
#
# class MockPhase(object):
#     def __init__(self):
#         self.name = "phase name"
#         self.phase_path = "phase_path"
#         self.search = MockOptimizer()
#         self.folders = [""]
#         self.phase_tag = ""
#
#     # noinspection PyUnusedLocal,PyMethodMayBeStatic
#     def run(self, *args, **kwargs):
#         return MockResult()
#
#
# @pytest.fixture(name="hyper_combined")
# def make_combined():
#     normal_phase = MockPhase()
#
#     # noinspection PyUnusedLocal
#     def run_hyper(*args, **kwargs):
#         return MockResult()
#
#     # noinspection PyTypeChecker
#     hyper_combined = extensions.CombinedHyperPhase(
#         normal_phase, hyper_phase_classes=(extensions.HyperNoisePhase,)
#     )
#
#     for phase in hyper_combined.hyper_phases:
#         phase.run_hyper = run_hyper
#
#     return hyper_combined
#
#
# class TestHyperAPI(object):
#     def test_combined_result(self, hyper_combined):
#         result = hyper_combined.run(datasets=None)
#
#         assert hasattr(result, "hyper_noise")
#         assert isinstance(result.hyper_noise, MockResult)
#
#         assert hasattr(result, "hyper_combined")
#         assert isinstance(result.hyper_combined, MockResult)
#
#     def test_combine_variables(self, hyper_combined):
#         result = MockResult()
#         hyper_noise_result = MockResult()
#
#         hyper_noise_result.model = af.ModelMapper()
#
#         hyper_noise_result.model.hyper_noise_scalar_of_ci_regions = (
#             ci_hyper.CIHyperNoiseScalar
#         )
#
#         result.hyper_noise = hyper_noise_result
#
#         model = hyper_combined.combine_variables(result)
#
#         assert isinstance(model.hyper_noise_scalar_of_ci_regions, af.PriorModel)
#
#         assert model.hyper_noise_scalar_of_ci_regions.cls == ci_hyper.CIHyperNoiseScalar
#
#     def test_instantiation(self, hyper_combined):
#         assert len(hyper_combined.hyper_phases) == 1
#
#         noise_phase = hyper_combined.hyper_phases[0]
#
#         assert noise_phase.hyper_name == "hyper_noise"
#         assert isinstance(noise_phase, extensions.HyperNoisePhase)
#
#     # def test_hyper_result(self, ccd_data_7x7):
#     #     normal_phase = MockPhase()
#     #
#     #     # noinspection PyTypeChecker
#     #     phase = extensions.HyperGalaxyPhase(normal_phase)
#     #
#     #     # noinspection PyUnusedLocal
#     #     def run_hyper(*args, **kwargs):
#     #         return MockResult()
#     #
#     #     phase.run_hyper = run_hyper
#     #
#     #     result = phase.run(ccd_data_7x7)
#     #
#     #     assert hasattr(result, "hyper_galaxy")
#     #     assert isinstance(result.hyper_galaxy, MockResult)
