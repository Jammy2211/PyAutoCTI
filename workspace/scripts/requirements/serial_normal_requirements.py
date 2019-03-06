from autofit.core import model_mapper as mm
from autocti.model import arctic_params as ap
import workspace_jam.scripts.requirements as req

import matplotlib.pyplot as plt

model_mapper = mm.ModelMapper(serial=ap.SerialThreeSpecies)
model_mapper.serial.well_fill_alpha = 0.0
model_mapper.serial.well_fill_gamma = 0.0

input_model = [1.0e-4, 0.58, 0.01, 0.03, 0.9, 0.8, 3.5, 20.0]
requirements = [1.0, 0.0000631, 0.0084, 0.0039, 0.000303, 0.0193, 0.0300, 0.0004]
sigma_limit = 2.0

most_likely_lr, most_probable_lr, lower_limits_lr, upper_limits_lr, lower_error_lr, upper_error_lr, error_pecision_lr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Serial_Normal_lr/')

most_likely_mr, most_probable_mr, lower_limits_mr, upper_limits_mr, lower_error_mr, upper_error_mr, error_pecision_mr = \
    req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model, sigma_limit=sigma_limit,
                                name='Results/Serial_Normal_mr/')

most_likely_hr, most_probable_hr, lower_limits_hr, upper_limits_hr, lower_error_hr, upper_error_hr, error_pecision_hr = \
  req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model,
                                                sigma_limit=sigma_limit, name='Results/Serial_Normal_hr/')

most_likely_x2, most_probable_x2, lower_limits_x2, upper_limits_x2, lower_error_x2, upper_error_x2,  error_pecision_x2 = \
req.setup_multinest_results(model_mapper=model_mapper, input_model=input_model,
                                               sigma_limit=sigma_limit, name='Results/Serial_Normal_x2/')

ylabels = ['d', r'$\beta$', r'$\rho_{1}$', r'$\rho_{2}$', r'$\rho_{3}$', r'$\tau_{1}$', r'$\tau_{2}$', r'$\tau_{3}$']

plt.figure(figsize=(20, 15))
plt.suptitle('Accuracy of Serial CTI Model', fontsize=20)

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.plot([496, 993, 1986, 3972], [input_model[i], input_model[i], input_model[i], input_model[i]], linestyle='--')
    print(i, most_probable_hr[i], (lower_error_hr[i], upper_error_hr[i]))
    plt.errorbar(x=[496, 993, 1986, 3972], y=[most_probable_lr[i], most_probable_mr[i],
                                              most_probable_hr[i], most_probable_x2[i]],
                 yerr=[[lower_error_lr[i], lower_error_mr[i], lower_error_hr[i], lower_error_x2[i]],
                       [upper_error_lr[i], upper_error_mr[i], upper_error_hr[i], upper_error_x2[i]]])
    plt.xlabel('Number of columns', fontsize=12)
    plt.ylabel(ylabels[i], fontsize=12)

plt.show()

plt.figure(figsize=(20, 15))
plt.suptitle('Precision of Serial CTI Model', fontsize=20)

for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.plot([496, 993, 1986, 3972], [error_pecision_lr[i], error_pecision_mr[i],
                                      error_pecision_hr[i], error_pecision_x2[i]])
    plt.xlabel('Number of columns', fontsize=12)
    plt.ylabel(ylabels[i], fontsize=12)
    # plt.plot([517, 1034, 2119, 4238], [error_pecision_lr[i], error_pecision_mr[i]])
                          #            error_pecision_hr[i], error_pecision_x2[i]])

plt.show()