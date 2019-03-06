from autofit.core import model_mapper as mm
from autocti.model import arctic_params as ap
import workspace_jam.scripts.requirements as req

import matplotlib.pyplot as plt

model_mapper = mm.ModelMapper(parallel=ap.ParallelTwoSpecies)
model_mapper.parallel.well_fill_alpha = 0.0
model_mapper.parallel.well_fill_gamma = 0.0

input_model = [10000.0, 0.58, 0.13, 0.25, 1.25, 4.4]
sigma_limit = 2.0

most_likely_lr, most_probable_lr, lower_limits_lr, upper_limits_lr, lower_error_lr, upper_error_lr, error_pecision_lr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_lr_pn/')

most_likely_mr, most_probable_mr, lower_limits_mr, upper_limits_mr, lower_error_mr, upper_error_mr, error_pecision_mr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Parallel_mr_pn/')

most_likely_hr, most_probable_hr, lower_limits_hr, upper_limits_hr, lower_error_hr, upper_error_hr, error_pecision_hr = \
req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model,
                                                sigma_limit=sigma_limit, name='Results/Parallel_hr_pn/')

most_likely_x2,  most_probable_x2, lower_limits_x2, upper_limits_x2, lower_error_x2, upper_error_x2,  error_pecision_x2 = \
req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model,
                                                sigma_limit=sigma_limit, name='Results/Parallel_x2_pn/')

ylabels = ['d', r'$\beta$', r'$\rho_{1}$', r'$\rho_{2}$', r'$\tau_{1}$', r'$\tau_{2}$']

plt.figure(figsize=(20, 15))
plt.suptitle('Accuracy of Parallel CTI Model', fontsize=20)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot([517, 1034, 2068, 4136], [input_model[i], input_model[i], input_model[i], input_model[i]], linestyle='--')
    plt.errorbar(x=[517, 1034, 2068, 4136], y=[most_probable_lr[i], most_probable_mr[i],
                                               most_probable_hr[i], most_probable_x2[i]],
                 yerr=[[lower_error_lr[i], lower_error_mr[i], lower_error_hr[i], lower_error_x2[i]],
                       [upper_error_lr[i], upper_error_mr[i], upper_error_hr[i], upper_error_x2[i]]])
    plt.xlabel('Number of columns', fontsize=12)
    plt.ylabel(ylabels[i], fontsize=12)

plt.show()

plt.figure(figsize=(20, 15))
plt.suptitle('Precision of Parallel CTI Model', fontsize=20)

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot([517, 1034, 2068, 4136], [error_pecision_lr[i], error_pecision_mr[i],
                                       error_pecision_hr[i], error_pecision_x2[i]])
    plt.xlabel('Number of columns', fontsize=12)
    plt.ylabel(ylabels[i], fontsize=12)

plt.show()