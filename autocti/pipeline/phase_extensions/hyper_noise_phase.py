# import copy
#
# import numpy as np
# from typing import cast
#
# import autofit as af
# from .hyper_phase import HyperPhase
#
# from autocti.charge_injection import ci_fit
#
#
# class HyperNoisePhase(HyperPhase):
#     def __init__(self, phase):
#
#         super().__init__(phase=phase, hyper_name="hyper_galaxy")
#
#     class Analysis(af.Analysis):
#         def __init__(
#             self, ci_datas_extracted, ci_datas_full, hyper_noise_scaling_map
#         ):
#             """
#             An analysis to fit the noise for a single galaxy image.
#             Parameters
#             ----------
#             lens_data: LensData
#                 Lens instrument, including an image and noise
#             hyper_noise_scaling_map: ndarray
#                 An image produce of the overall system by a model
#             hyper_galaxy_image_1d_path_dict: ndarray
#                 The contribution of one galaxy to the model image
#             """
#
#             self.ci_datas_extracted = ci_datas_extracted
#             self.ci_datas_full = ci_datas_full
#
#             self.hyper_noise_scaling_map = hyper_noise_scaling_map
#
#             self.plot_hyper_galaxy_subplot = af.conf.instance.visualize.get(
#                 "plots", "plot_hyper_galaxy_subplot", bool
#             )
#
#         def visualize(self, instance, image_path, during_analysis):
#
#             pass
#
#         def fit(self, instance):
#             """
#             Fit the model image to the real image by scaling the hyper_galaxy noise.
#             Parameters
#             ----------
#             instance: ModelInstance
#                 A model instance with a hyper_galaxy galaxy property
#             Returns
#             -------
#             fit: float
#             """
#
#             fit = self.fit_for_hyper_noise_scalars(
#                 hyper_noise_scalars=instance.hyper_galaxy,
#             )
#
#             return fit.figure_of_merit
#
#         def fit_for_hyper_noise_scalars(
#             self, hyper_noise_scalars,
#         ):
#
#             hyper_noise_map = ci_fit.hyper_noise_map_from_noise_map_and_noise_scalings(
#                 noise_scaling_maps=self.masked_hyper_ci_data.noise_scaling_maps,
#                 hyper_noise_scalars=hyper_noise_scalars,
#                 noise_map=self.masked_hyper_ci_data.noise_map,
#             )
#
#             image_1d = lens_fit.image_1d_from_lens_data_and_hyper_image_sky(
#                 lens_data=self.lens_data, hyper_image_sky=hyper_image_sky
#             )
#
#             if hyper_background_noise is not None:
#                 noise_map_1d = hyper_background_noise.noise_map_scaled_noise_from_noise_map(
#                     noise_map=self.lens_data.noise_map_1d
#                 )
#             else:
#                 noise_map_1d = self.lens_data.noise_map_1d
#
#             hyper_noise_map_1d = hyper_noise_scalars.hyper_noise_map_from_hyper_images_and_noise_map(
#                 hyper_model_image=self.hyper_noise_scaling_map,
#                 hyper_galaxy_image=self.hyper_galaxy_image_1d_path_dict,
#                 noise_map=self.lens_data.noise_map_1d,
#             )
#
#             noise_map_1d = noise_map_1d + hyper_noise_map_1d
#             if self.lens_data.hyper_noise_map_max is not None:
#                 noise_map_1d[
#                     noise_map_1d > self.lens_data.hyper_noise_map_max
#                 ] = self.lens_data.hyper_noise_map_max
#
#             return lens_fit.LensDataFit(
#                 image_1d=image_1d,
#                 noise_map_1d=noise_map_1d,
#                 mask_1d=self.lens_data.mask_1d,
#                 model_image_1d=self.hyper_noise_scaling_map,
#                 grid_stack=self.lens_data.grid_stack,
#             )
#
#         @classmethod
#         def describe(cls, instance):
#             return "Running hyper_galaxy galaxy fit for HyperGalaxy:\n{}".format(
#                 instance.hyper_galaxy
#             )
#
#     def run_hyper(self, data, results=None):
#         """
#         Run a fit for each galaxy from the previous phase.
#         Parameters
#         ----------
#         data: LensData
#         results: ResultsCollection
#             Results from all previous phases
#         Returns
#         -------
#         results: HyperGalaxyResults
#             A collection of results, with one item per a galaxy
#         """
#         phase = self.make_hyper_phase()
#
#         lens_data = ld.LensData(
#             ccd_data=data,
#             mask=results.last.mask_2d,
#             sub_grid_size=cast(phase_imaging.PhaseImaging, phase).sub_grid_size,
#             image_psf_shape=cast(phase_imaging.PhaseImaging, phase).image_psf_shape,
#             positions=results.last.positions,
#             interp_pixel_scale=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).interp_pixel_scale,
#             cluster_pixel_scale=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).cluster_pixel_scale,
#             cluster_pixel_limit=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).cluster_pixel_limit,
#             uses_inversion=cast(phase_imaging.PhaseImaging, phase).uses_inversion,
#             uses_cluster_inversion=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).uses_cluster_inversion,
#             hyper_noise_map_max=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).hyper_noise_map_max,
#         )
#
#         model_image_1d = results.last.hyper_model_image_1d
#         hyper_galaxy_image_1d_path_dict = results.last.hyper_galaxy_image_1d_path_dict
#
#         hyper_result = copy.deepcopy(results.last)
#         hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
#             hyper_result.constant
#         )
#         hyper_result.analysis.uses_hyper_images = True
#         hyper_result.analysis.hyper_model_image_1d = model_image_1d
#         hyper_result.analysis.hyper_galaxy_image_1d_path_dict = (
#             hyper_galaxy_image_1d_path_dict
#         )
#
#         for path, galaxy in results.last.path_galaxy_tuples:
#
#             optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])
#
#             optimizer.phase_tag = ""
#
#             # TODO : This is a HACK :O
#
#             optimizer.variable.galaxies = []
#
#             optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
#             )
#             optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
#             )
#             optimizer.n_live_points = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_n_live_points", int
#             )
#
#             optimizer.variable.hyper_galaxy = g.HyperGalaxy
#
#             if self.include_sky_background:
#                 optimizer.variable.hyper_image_sky = hd.HyperImageSky
#
#             if self.include_noise_background:
#                 optimizer.variable.hyper_background_noise = hd.HyperBackgroundNoise
#
#             # If array is all zeros, galaxy did not have image in previous phase and
#             # should be ignored
#             if not np.all(hyper_galaxy_image_1d_path_dict[path] == 0):
#
#                 analysis = self.Analysis(
#                     lens_data=lens_data,
#                     hyper_noise_scaling_map=model_image_1d,
#                     hyper_galaxy_image_1d_path_dict=hyper_galaxy_image_1d_path_dict[
#                         path
#                     ],
#                 )
#
#                 result = optimizer.fit(analysis)
#
#                 def transfer_field(name):
#                     if hasattr(result.constant, name):
#                         setattr(
#                             hyper_result.constant.object_for_path(path),
#                             name,
#                             getattr(result.constant, name),
#                         )
#                         setattr(
#                             hyper_result.variable.object_for_path(path),
#                             name,
#                             getattr(result.variable, name),
#                         )
#
#                 transfer_field("hyper_galaxy")
#
#                 hyper_result.constant.hyper_image_sky = getattr(
#                     result.constant, "hyper_image_sky"
#                 )
#                 hyper_result.variable.hyper_image_sky = getattr(
#                     result.variable, "hyper_image_sky"
#                 )
#
#                 hyper_result.constant.hyper_background_noise = getattr(
#                     result.constant, "hyper_background_noise"
#                 )
#                 hyper_result.variable.hyper_background_noise = getattr(
#                     result.variable, "hyper_background_noise"
#                 )
#
#         return hyper_result
#
#
# class HyperGalaxyBackgroundSkyPhase(HyperNoisePhase):
#     def __init__(self, phase):
#         super().__init__(phase=phase)
#         self.include_sky_background = True
#         self.include_noise_background = False
#
#
# class HyperGalaxyBackgroundNoisePhase(HyperNoisePhase):
#     def __init__(self, phase):
#         super().__init__(phase=phase)
#         self.include_sky_background = False
#         self.include_noise_background = True
#
#
# class HyperGalaxyBackgroundBothPhase(HyperNoisePhase):
#     def __init__(self, phase):
#         super().__init__(phase=phase)
#         self.include_sky_background = True
#         self.include_noise_background = True
#
#
# class HyperGalaxyAllPhase(HyperPhase):
#     def __init__(
#         self, phase, include_sky_background=False, include_noise_background=False
#     ):
#         super().__init__(phase=phase, hyper_name="hyper_galaxy")
#         self.include_sky_background = include_sky_background
#         self.include_noise_background = include_noise_background
#
#     def run_hyper(self, data, results=None):
#         """
#         Run a fit for each galaxy from the previous phase.
#         Parameters
#         ----------
#         data: LensData
#         results: ResultsCollection
#             Results from all previous phases
#         Returns
#         -------
#         results: HyperGalaxyResults
#             A collection of results, with one item per a galaxy
#         """
#         phase = self.make_hyper_phase()
#
#         lens_data = ld.LensData(
#             ccd_data=data,
#             mask=results.last.mask_2d,
#             sub_grid_size=cast(phase_imaging.PhaseImaging, phase).sub_grid_size,
#             image_psf_shape=cast(phase_imaging.PhaseImaging, phase).image_psf_shape,
#             positions=results.last.positions,
#             interp_pixel_scale=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).interp_pixel_scale,
#             cluster_pixel_scale=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).cluster_pixel_scale,
#             cluster_pixel_limit=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).cluster_pixel_limit,
#             uses_inversion=cast(phase_imaging.PhaseImaging, phase).uses_inversion,
#             uses_cluster_inversion=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).uses_cluster_inversion,
#             hyper_noise_map_max=cast(
#                 phase_imaging.PhaseImaging, phase
#             ).hyper_noise_map_max,
#         )
#
#         model_image_1d = results.last.hyper_model_image_1d
#         hyper_galaxy_image_1d_path_dict = results.last.hyper_galaxy_image_1d_path_dict
#
#         hyper_result = copy.deepcopy(results.last)
#         hyper_result.variable = hyper_result.variable.copy_with_fixed_priors(
#             hyper_result.constant
#         )
#         hyper_result.analysis.uses_hyper_images = True
#         hyper_result.analysis.hyper_model_image_1d = model_image_1d
#         hyper_result.analysis.hyper_galaxy_image_1d_path_dict = (
#             hyper_galaxy_image_1d_path_dict
#         )
#
#         for path, galaxy in results.last.path_galaxy_tuples:
#
#             optimizer = phase.optimizer.copy_with_name_extension(extension=path[-1])
#
#             optimizer.phase_tag = ""
#
#             # TODO : This is a HACK :O
#
#             optimizer.variable.galaxies = []
#             optimizer.variable.galaxies = []
#             optimizer.variable.galaxies = []
#
#             optimizer.const_efficiency_mode = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_const_efficiency_mode", bool
#             )
#             optimizer.sampling_efficiency = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_sampling_efficiency", float
#             )
#             optimizer.n_live_points = af.conf.instance.non_linear.get(
#                 "MultiNest", "extension_hyper_galaxy_n_live_points", int
#             )
#
#             optimizer.variable.hyper_galaxy = g.HyperGalaxy
#
#             if self.include_sky_background:
#                 optimizer.variable.hyper_image_sky = hd.HyperImageSky
#
#             if self.include_noise_background:
#                 optimizer.variable.hyper_background_noise = hd.HyperBackgroundNoise
#
#             # If array is all zeros, galaxy did not have image in previous phase and
#             # should be ignored
#             if not np.all(hyper_galaxy_image_1d_path_dict[path] == 0):
#
#                 analysis = self.Analysis(
#                     lens_data=lens_data,
#                     model_image_1d=model_image_1d,
#                     galaxy_image_1d=hyper_galaxy_image_1d_path_dict[path],
#                 )
#
#                 result = optimizer.fit(analysis)
#
#                 def transfer_field(name):
#                     if hasattr(result.constant, name):
#                         setattr(
#                             hyper_result.constant.object_for_path(path),
#                             name,
#                             getattr(result.constant, name),
#                         )
#                         setattr(
#                             hyper_result.variable.object_for_path(path),
#                             name,
#                             getattr(result.variable, name),
#                         )
#
#                 transfer_field("hyper_galaxy")
#                 transfer_field("hyper_image_sky")
#                 transfer_field("hyper_background_noise")
#
#         return hyper_result
