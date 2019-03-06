from autofit.core import model_mapper as mm
from autocti.model import arctic_params as ap
import workspace_jam.scripts.requirements as req
import workspace_jam.scripts.requirements.requirement_converter as convert

import matplotlib.pyplot as plt
import numpy as np

model_mapper = mm.ModelMapper(parallel=ap.ParallelTwoSpecies)
model_mapper.parallel.well_fill_alpha = 0.0
model_mapper.parallel.well_fill_gamma = 0.0
sigma_limit = 2.0

input_model = [10000.0, 0.58, 0.13, 0.25, 1.25, 4.4]
input_rho = np.array([0.13, 0.25])
input_tau = np.array([1.25, 4.4])
delta_ellipticity_true = convert.convert_to_ellipticity(rho=input_rho, tau=input_tau)

print('Model True')
print(delta_ellipticity_true)
print(np.sum(delta_ellipticity_true))
print()


most_likely_lr, most_probable_lr, lower_limits_lr, upper_limits_lr, lower_error_lr, upper_error_lr, error_pecision_lr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_lr_pn/')

model_rho_lr = np.array([most_probable_lr[2], most_probable_lr[3]])
model_tau_lr = np.array([most_probable_lr[4], most_probable_lr[5]])
delta_ellipticity_model_lr = convert.convert_to_ellipticity(rho=model_rho_lr, tau=model_tau_lr)

upper_rho_lr = np.array([upper_limits_lr[2], upper_limits_lr[3]])
upper_tau_lr = np.array([upper_limits_lr[4], upper_limits_lr[5]])
delta_ellipticity_upper_lr = convert.convert_to_ellipticity(rho=upper_rho_lr, tau=upper_tau_lr)

lower_rho_lr = np.array([lower_limits_lr[2], lower_limits_lr[3]])
lower_tau_lr = np.array([lower_limits_lr[4], lower_limits_lr[5]])
delta_ellipticity_lower_lr = convert.convert_to_ellipticity(rho=lower_rho_lr, tau=lower_tau_lr)

print('Induced Delta Elliptcitiy / Requirement')
print((np.sum(delta_ellipticity_model_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_upper_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_lower_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print()

ellipticity_requirement_lr = (np.sum(delta_ellipticity_model_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_upper_lr = (np.sum(delta_ellipticity_upper_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_lower_lr = (np.sum(delta_ellipticity_lower_lr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_error_upper_lr = ellipticity_requirement_upper_lr - ellipticity_requirement_lr
ellipticity_error_lower_lr = ellipticity_requirement_lr - ellipticity_requirement_lower_lr

most_likely_mr, most_probable_mr, lower_limits_mr, upper_limits_mr, lower_error_mr, upper_error_mr, error_pecision_mr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_mr_pn/')

model_rho_mr = np.array([most_probable_mr[2], most_probable_mr[3]])
model_tau_mr = np.array([most_probable_mr[4], most_probable_mr[5]])
delta_ellipticity_model_mr = convert.convert_to_ellipticity(rho=model_rho_mr, tau=model_tau_mr)

upper_rho_mr = np.array([upper_limits_mr[2], upper_limits_mr[3]])
upper_tau_mr = np.array([upper_limits_mr[4], upper_limits_mr[5]])
delta_ellipticity_upper_mr = convert.convert_to_ellipticity(rho=upper_rho_mr, tau=upper_tau_mr)

lower_rho_mr = np.array([lower_limits_mr[2], lower_limits_mr[3]])
lower_tau_mr = np.array([lower_limits_mr[4], lower_limits_mr[5]])
delta_ellipticity_lower_mr = convert.convert_to_ellipticity(rho=lower_rho_mr, tau=lower_tau_mr)

print('Induced Delta Elliptcitiy / Requirement')
print((np.sum(delta_ellipticity_model_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_upper_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_lower_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print()

ellipticity_requirement_mr = (np.sum(delta_ellipticity_model_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_upper_mr = (np.sum(delta_ellipticity_upper_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_lower_mr = (np.sum(delta_ellipticity_lower_mr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_error_upper_mr = ellipticity_requirement_upper_mr - ellipticity_requirement_mr
ellipticity_error_lower_mr = ellipticity_requirement_mr - ellipticity_requirement_lower_mr

most_likely_hr, most_probable_hr, lower_limits_hr, upper_limits_hr, lower_error_hr, upper_error_hr, error_pecision_hr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_hr_pn/')

model_rho_hr = np.array([most_probable_hr[2], most_probable_hr[3]])
model_tau_hr = np.array([most_probable_hr[4], most_probable_hr[5]])
delta_ellipticity_model_hr = convert.convert_to_ellipticity(rho=model_rho_hr, tau=model_tau_hr)

upper_rho_hr = np.array([upper_limits_hr[2], upper_limits_hr[3]])
upper_tau_hr = np.array([upper_limits_hr[4], upper_limits_hr[5]])
delta_ellipticity_upper_hr = convert.convert_to_ellipticity(rho=upper_rho_hr, tau=upper_tau_hr)

lower_rho_hr = np.array([lower_limits_hr[2], lower_limits_hr[3]])
lower_tau_hr = np.array([lower_limits_hr[4], lower_limits_hr[5]])
delta_ellipticity_lower_hr = convert.convert_to_ellipticity(rho=lower_rho_hr, tau=lower_tau_hr)

ellipticity_requirement_hr = (np.sum(delta_ellipticity_model_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_upper_hr = (np.sum(delta_ellipticity_upper_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_lower_hr = (np.sum(delta_ellipticity_lower_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_error_upper_hr = ellipticity_requirement_upper_hr - ellipticity_requirement_hr
ellipticity_error_lower_hr = ellipticity_requirement_hr - ellipticity_requirement_lower_hr

print('Induced Delta Elliptcitiy / Requirement')
print((np.sum(delta_ellipticity_model_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_upper_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_lower_hr) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print()


most_likely_x2, most_probable_x2, lower_limits_x2, upper_limits_x2, lower_error_x2, upper_error_x2, error_pecision_x2 = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_x2_pn/')

model_rho_x2 = np.array([most_probable_x2[2], most_probable_x2[3]])
model_tau_x2 = np.array([most_probable_x2[4], most_probable_x2[5]])
delta_ellipticity_model_x2 = convert.convert_to_ellipticity(rho=model_rho_x2, tau=model_tau_x2)

upper_rho_x2 = np.array([upper_limits_x2[2], upper_limits_x2[3]])
upper_tau_x2 = np.array([upper_limits_x2[4], upper_limits_x2[5]])
delta_ellipticity_upper_x2 = convert.convert_to_ellipticity(rho=upper_rho_x2, tau=upper_tau_x2)

lower_rho_x2 = np.array([lower_limits_x2[2], lower_limits_x2[3]])
lower_tau_x2 = np.array([lower_limits_x2[4], lower_limits_x2[5]])
delta_ellipticity_lower_x2 = convert.convert_to_ellipticity(rho=lower_rho_x2, tau=lower_tau_x2)

ellipticity_requirement_x2 = (np.sum(delta_ellipticity_model_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_upper_x2 = (np.sum(delta_ellipticity_upper_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_requirement_lower_x2 = (np.sum(delta_ellipticity_lower_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4)
ellipticity_error_upper_x2 = ellipticity_requirement_upper_x2 - ellipticity_requirement_x2
ellipticity_error_lower_x2 = ellipticity_requirement_x2 - ellipticity_requirement_lower_x2

print('Induced Delta Elliptcitiy / Requirement')
print((np.sum(delta_ellipticity_model_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_upper_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print((np.sum(delta_ellipticity_lower_x2) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print()

plt.plot([517, 1034, 2068, 4136], [1.0, 1.0, 1.0, 1.0], linestyle='--', color='r')
plt.plot([517, 1034, 2068, 4136], [-1.0, -1.0, -1.0, -1.0], linestyle='--', color='r')
plt.errorbar(x=[517, 1034, 2068, 4136], y=[ellipticity_requirement_lr, ellipticity_requirement_mr,
                                   ellipticity_requirement_hr, ellipticity_requirement_x2],
             yerr=[[ellipticity_error_lower_lr, ellipticity_error_lower_mr, 
                    ellipticity_error_lower_hr, ellipticity_error_lower_x2],
                   [ellipticity_error_upper_lr, ellipticity_error_upper_mr,
                    ellipticity_error_upper_hr, ellipticity_error_upper_x2]], color='b', ecolor='b')
plt.xlabel('Number of columns', fontsize=12)
plt.ylabel('Delta Ellipticity / Requirement (1.1e-4)', fontsize=12)
plt.show()