from autofit.core import model_mapper as mm
from autofit.core import non_linear as nl
from autocti.model import arctic_params as ap
from workspace.pipelines import parallel_x3

import matplotlib.pyplot as plt

def setup_multinest_results(model_mapper, name, input_model, sigma_limit=2.0, requirements=None):

    multinest = nl.MultiNest(model_mapper=model_mapper, name=name, sigma_limit=sigma_limit)

    most_likely = multinest.most_likely_from_summary()
    most_probable = multinest.most_probable_from_summary()
    upper_limits = multinest.model_at_upper_sigma_limit(sigma_limit=sigma_limit)
    lower_limits = multinest.model_at_lower_sigma_limit(sigma_limit=sigma_limit)

    upper_error = list(map(lambda mp, upper : upper - mp, most_probable, upper_limits))
    lower_error = list(map(lambda mp, lower : mp - lower, most_probable, lower_limits))
    errors = multinest.model_errors_at_sigma_limit(sigma_limit=sigma_limit)
    error_precision = list(map(lambda upper, lower : upper - lower, upper_limits, lower_limits))

    most_probable_difference = list(map(lambda input, mp :  mp - input, input_model, most_probable))
    upper_limit_difference = list(map(lambda input, upper : upper - input, input_model, upper_limits))
    lower_limit_difference = list(map(lambda input, lower : lower - input, input_model, lower_limits))

    # print('Model Accuracy')
    #
    # for i in range(multinest.variable.prior_count):
    #    print()
    #    print('input = ' + str(input_model[i]) + '  most probable = ' + str(most_probable[i]) +
    #          ' (' + str(lower_limits[i]) + ', ' + str(upper_limits[i]) + ')')
    #    print('error precision = ' + str(error_precision[i]))
    #    print(str(lower_limit_difference[i]) + ' ' + str(most_probable_difference[i]) + ' ' + str(upper_limit_difference[i]))

    #    if lower_limits[i] > most_probable[i]:
    #        stop
    #
    # print()
    # print('Model Requirements')

    # print()
    # print(list(map(lambda requirement, error : error / requirement, requirements, errors)))

    return most_likely, most_probable, lower_limits, upper_limits, upper_error, lower_error, error_precision