import os

from autofit import conf
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_pattern
from autocti.charge_injection import ci_data

from autocti.model import arctic_settings

from multiprocessing import Pool

# Welcome to the pipeline runner. This tool allows you to CTI calibration data on strong lenses, and pass it to
# pipelines for a PyAutoCTI analysis. To show you around, we'll load up some example data and run it through some of
# the example pipelines that come distributed with PyAutoCTI.

# The runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out data-set names
# and pipelines to perform different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

# The pipeline runner is fairly self explanatory. Make sure to checkout the pipelines in the
#  workspace/pipelines/examples/ folder - they come with detailed descriptions of what they do. I hope that you'll
# expand on them for your own personal scientific needs

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

# It is convenient to specify the image type and model used to simulate that image as strings, so that if the
# pipeline is applied to multiple images we don't have to change all of the path entries in the
# load_ci_data_from_fits function below.

ci_data_name = 'ci_x2_images_uniform_high_res' # Charge injection data consisting of 2 images with uniform injections.
ci_data_model = 'serial_x1_species' # Shows the data was creating using a serial CTI model with one species.

# The shape of the charge injection images, which is required to set up their charge injection regions
shape = (2316, 2119)

# The charge injection regions on the CCD, which in this case is 7 equally spaced rectangular blocks.
ci_regions = [(0, 30, 51, shape[1]-20), (330, 360, 51, shape[1]-20),
              (660, 690, 51, shape[1]-20), (990, 1020, 51, shape[1]-20),
              (1320, 1350, 51, shape[1]-20), (1650, 1680, 51, shape[1]-20),
              (1980, 2010, 51, shape[1]-20)]

# The normalization of the charge injection for each image.
normalizations=[1000.0, 84700.0]

# Create the charge injection pattern objects used for this pipeline.
patterns = ci_pattern.uniform_from_lists(normalizations=normalizations, regions=ci_regions)

# The frame geometry of the charge injection images we are fitting.
frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

# There are 8 images to load, which is what the normalizations list tells us. To load them, its easiest to create a
# 'data' list and then iterate over a for loop, to append each set of data to the list of data we pass to the pipeline
# when we run it.

# We create the path to the data itself inside this folder, based on the data name and model.
data_path = path +'/data/' + ci_data_name + '/' + ci_data_model

datas = []

for image_index in range(len(normalizations)):

    datas.append(ci_data.load_ci_data_from_fits(
                 frame_geometry=frame_geometry, ci_pattern=patterns[image_index],
                 ci_image_path=data_path+'/ci_image_' + str(image_index) + '.fits',
                 ci_pre_cti_path=data_path+'/ci_pre_cti_' + str(image_index) + '.fits',
                 ci_noise_map_from_single_value=4.0))

# The CTI settings of arCTIc, which models the CCD read-out including CTI. For serial ci data, we include 'charge
# injection mode' which accounts for the fact that every pixel is transferred over the full CCD.
serial_cti_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                        charge_injection_mode=False, readout_offset=0)
cti_settings = arctic_settings.ArcticSettings(serial=serial_cti_settings)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the ci data to its run function.
# Below, we'll' use a 2 phase example pipeline to fit the data with a one species serial CTI model.
# Checkout workspace/pipelines/examples/serial_x1_species.py' for a full description of the pipeline.

# The pool command tells our code to serialize the analysis over 2 CPU's, where each CPU fits a different charge
# injection image

from workspace_jam.pipelines import serial_x3_species
pipeline = serial_x3_species.make_pipeline(pipeline_path=ci_data_name + '/')
pipeline.run(ci_datas=datas, cti_settings=cti_settings, pool=Pool(processes=2))