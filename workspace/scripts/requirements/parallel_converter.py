from autofit.core import model_mapper as mm
from autocti.model import arctic_params as ap
import workspace_jam.scripts.requirements as req
import workspace_jam.scripts.requirements.requirement_converter as convert

import numpy as np

model_mapper = mm.ModelMapper(parallel=ap.ParallelTwoSpecies)
model_mapper.parallel.well_fill_alpha = 0.0
model_mapper.parallel.well_fill_gamma = 0.0

input_model = [10000.0, 0.58, 0.13, 0.25, 1.25, 4.4]
input_rho = np.array([0.13, 0.25])
input_tau = np.array([1.25, 4.4])
sigma_limit = 2.0

most_probable, lower_limits, upper_limits, lower_error, upper_error, error_pecision = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_mr_pn/')

delta_ellipticity_true = convert.convert_to_ellipticity(rho=input_rho, tau=input_tau)

model_rho = np.array([most_probable[2], most_probable[3]])
model_tau = np.array([most_probable[4], most_probable[5]])
delta_ellipticity_model = convert.convert_to_ellipticity(rho=model_rho, tau=model_tau)

upper_rho = np.array([upper_limits[2], upper_limits[3]])
upper_tau = np.array([upper_limits[4], upper_limits[5]])
delta_ellipticity_upper = convert.convert_to_ellipticity(rho=upper_rho, tau=upper_tau)

lower_rho = np.array([lower_limits[2], lower_limits[3]])
lower_tau = np.array([lower_limits[4], lower_limits[5]])
delta_ellipticity_lower = convert.convert_to_ellipticity(rho=lower_rho, tau=lower_tau)

print('Model True')
print(delta_ellipticity_true)
print(np.sum(delta_ellipticity_true))
print()

print('Model most probable')
print(delta_ellipticity_model)
print(np.sum(delta_ellipticity_model))
print()

print('Model upper')
print(delta_ellipticity_upper)
print(np.sum(delta_ellipticity_upper))
print()

print('Model lower')
print(delta_ellipticity_lower)
print(np.sum(delta_ellipticity_lower))
print()

print('Induced Delta Elliptcitiy')
print(np.abs(np.sum(delta_ellipticity_model) - np.sum(delta_ellipticity_true)))
print(np.abs(np.sum(delta_ellipticity_upper) - np.sum(delta_ellipticity_true)))
print(np.abs(np.sum(delta_ellipticity_lower) - np.sum(delta_ellipticity_true)))
print()

print('Induced Delta Elliptcitiy / Requirement')
print((np.sum(delta_ellipticity_model) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_upper) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_lower) - np.sum(delta_ellipticity_true)) / (1.1e-4))