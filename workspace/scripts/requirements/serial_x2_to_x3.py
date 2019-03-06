import workspace_jam.scripts.requirements.requirement_converter as convert

import numpy as np

input_model = [0.01, 0.03, 0.9, 0.8, 3.5, 20.0, 1.0e-4, 0.58]
input_rho = np.array([0.01, 0.03, 0.9])
input_tau = np.array([0.8, 3.5, 20.0])

most_probable = np.array([0.024758, 0.88427, 1.28204, 18.6937, 3.5, 9.58])
upper_limits = np.array([0.04782, 0.9442, 2.5276, 21.8204, 29.4351, 0.5980])
lower_limits = np.array([0.0144, 0.84705, 0.69958, 17.2769, -31.104, 0.55735])

sigma_limit = 2.0

delta_ellipticity_true = convert.convert_to_ellipticity(rho=input_rho, tau=input_tau)

model_rho = np.array([most_probable[0], most_probable[1]])
model_tau = np.array([most_probable[2], most_probable[3]])
delta_ellipticity_model = convert.convert_to_ellipticity(rho=model_rho, tau=model_tau)

upper_rho = np.array([upper_limits[0], upper_limits[1]])
upper_tau = np.array([upper_limits[2], upper_limits[3]])
delta_ellipticity_upper = convert.convert_to_ellipticity(rho=upper_rho, tau=upper_tau)

lower_rho = np.array([lower_limits[0], lower_limits[1]])
lower_tau = np.array([lower_limits[2], lower_limits[3]])
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
print(np.abs(np.sum(delta_ellipticity_model) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print(np.abs(np.sum(delta_ellipticity_upper) - np.sum(delta_ellipticity_true)) / (1.1e-4))
print(np.abs(np.sum(delta_ellipticity_lower) - np.sum(delta_ellipticity_true)) / (1.1e-4))