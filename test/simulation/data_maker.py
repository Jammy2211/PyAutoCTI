from test.simulation import makers

# Welcome to the PyAutoCTI test suite data maker. Here, we'll make the suite of data that we use to test and profile
# PyAutoCTI. This consists of the following sets of images:

# A uniform charge injectiono line image, generated using a parallel CTI model with 1 trap species.

# Each image is generated at 4 resolutions, 36x36 (Patch), 120x120 (Low_Res), 300x300 (Mid_Res), 600x600 (High_Res).

# To simulate each lens, we pass it a name and call its maker. In the makers.py file, you'll see these functions.
makers.make_ci_uniform_parallel_x1_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_serial_x1_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_parallel_and_serial_x1_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_parallel_x3_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_serial_x3_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_parallel_and_serial_x3_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_cosmic_rays_parallel_x1_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_cosmic_rays_serial_x1_species(data_resolutions=['patch'], normalizations=[1000.0, 84700.0])
makers.make_ci_uniform_cosmic_rays_parallel_and_serial_x1_species(data_resolutions=['patch'],
                                                                  normalizations=[1000.0, 84700.0])