from test_autocti.simulators import simulators

"""
Welcome to the PyAutoCTI dataset generator. Here, we'll make the datasets used to test and profile PyAutoCTI.

This consists of the following sets of images:

- An image where CTI is added with 1 parallel trap species.
- An image where CTI is added with 3 parallel trap species.
- An image where CTI is added with 1 serial trap species.
- An image where CTI is added with 3 serial trap species.
- An image where CTI is added with 1 parallel trap species and 1 serial trap species.
- An image where CTI is added with 3 parallel trap species and 3 serial trap species.

Each image is generated at 4 resolutions, 36x36 (Patch), 120x120 (Low_Res), 300x300 (Mid_Res), 600x600 (High_Res).
"""

resolutions = ["patch"]
normalizations = [1000.0, 10000.0, 25000.0, 84700.0]

# To simulate each galaxy, we pass it a resolution nd call its simulate function.

for resolution in resolutions:

    simulators.simulate__ci_uniform__parallel_x1(
        resolution=resolution, normalizations=normalizations
    )
    simulators.simulate__ci_uniform__parallel_x3(
        resolution=resolution, normalizations=normalizations
    )
    simulators.simulate__ci_uniform__serial_x1(
        resolution=resolution, normalizations=normalizations
    )
    simulators.simulate__ci_uniform__serial_x3(
        resolution=resolution, normalizations=normalizations
    )
    simulators.simulate__ci_uniform__parallel_x1__serial_x1(
        resolution=resolution, normalizations=normalizations
    )
    simulators.simulate__ci_uniform__parallel_x3__serial_x3(
        resolution=resolution, normalizations=normalizations
    )
