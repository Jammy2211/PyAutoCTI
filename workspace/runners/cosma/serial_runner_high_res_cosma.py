from autofit import conf
from autocti.charge_injection import ci_frame
from autocti.charge_injection import ci_data
from autocti.charge_injection import ci_pattern

from autocti.model import arctic_settings

from multiprocessing import Pool

import os
import sys

### NOTE - if you have not already, complete the setup in 'workspace/runners/cosma/setup' before continuing with this
### cosma pipeline script.

# Welcome to the Cosma pipeline runner. Hopefully, you're familiar with runners at this point, and have been using them
# with PyAutoCTI to model CTI on your laptop. If not, I'd recommend you get used to doing that, before trying to
# run PyAutoCTI on a super-computer. You need some familiarity with the software and before trying to model a large
# amount of charge injection imaging on a supercomputer!

# If you are ready, then let me take you through the Cosma runner. It is remarkably similar to the ordinary pipeline
# runners you're used to, however it makes a few changes for running jobs on cosma:

# 1) The data path is over-written to the path '/cosma5/data/durham/cosma_username/autocti/data' as opposed to the
#    workspace. As we saw in the setup, on cosma we don't store our data in our workspace.

# 2) The output path is over-written to the path '/cosma5/data/durham/cosma_username/autocti/output' as opposed to
#    the workspace. This is for the same reason as the data.

# Given your username is where your data is stored, you'll need to put your cosma username here.
cosma_username = 'pdtw24'

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path, and override the output path with the Cosma path.
conf.instance = conf.Config(config_path=path+'config',
                            output_path='/cosma5/data/durham/'+cosma_username+'/autocti/output')

# Lets take a look at a Cosma batch script, which can be found at 'workspace/runners/cosma/batch/pipeline_runner_cosma'.
# When we submit a PyAutoLens job to Cosma, we submit a 'batch' of jobs, whereby each job will run on one CPU of Cosma.
# Thus, if our lens sample contains, lets say, 4 lenses, we'd submit 4 jobs at the same time where each job applies
# our pipeline to each image.

# The fifth line of this batch script - '#SBATCH --array=1-4' is what species this. Its telling Cosma we're going to
# run 4 jobs, and the id's of those jobs will be numbered from 1 to 4. Infact, these ids are passed to this runner,
# and we'll use them to ensure that each jobs loads a different image. Lets get the cosma array id for our job.
batch_id = int(sys.argv[1])
# cosma_array_id = int(sys.argv[2])

# For a given COSMA run, we will assume all of the charge injection data-sets have identical properties. That is, they
# all have the same dimensions (shape), charge injection regions (ci_regions), charge injection normalizations
# (normalizations), charge injection patterns (patterns) and frame geometries defining the direction clocking and CTI
# (frame_geometry).

# If you need to change these value for a COSMA run, I recommend making a different runner script.

ci_data_name = 'ci_x8_images_uniform' # Charge injection data consisting of 2 images with uniform injections.
ci_data_resolution = 'high_res' # The resolution of the image.
ci_data_model = 'serial_x3_species' # Shows the data was creating using a serial CTI model with one species.
shape = (2316, 2119)

# The charge injection regions on the CCD, which in this case is 7 equally spaced rectangular blocks.
ci_regions = [(0, shape[0], 51, shape[1]-20)]

# The normalization of the charge injection for each image.
normalizations=[100.0, 500.0, 1000.0, 5000.0, 10000.0, 25000.0, 50000.0, 84700.0]

# Create the charge injection pattern objects used for this pipeline.
patterns = ci_pattern.uniform_from_lists(normalizations=normalizations, regions=ci_regions)

# The frame geometry of the charge injection images we are fitting.
frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()

# Now, I just want to really drive home what the above line is doing. For every job we run on Cosma, the cosma_array_id
# will be different. That is, job 1 will get a cosma_array_id of 1, job 2 will get an id of 2, and so on. This is our
# only unique identifier of every job, thus its our only hope of specifying for each job which image they load!

# Fortunately, we're used to specifying the data name as a string, so that our pipeline can be applied to multiple
# images with ease. On Cosma, we can apply the same logic, but put these strings in a list such that each Cosma job
# loads a different lens name based on its ID. neat, huh?

data_name = 'ci_x8_images_non_uniform/' + ci_data_resolution

ci_data_name = []
ci_data_name.append('') # Task number beings at 1, so keep index 0 blank
ci_data_name.append(data_name+'/serial_x3_species_model_4_1') # Index 1
ci_data_name.append(data_name+'/serial_x3_species_model_4_2') # Index 2
ci_data_name.append(data_name+'/serial_x3_species_model_4_3') # Index 3
ci_data_name.append(data_name+'/serial_x3_species_model_5') # Index 4
ci_data_name.append(data_name+'/serial_x3_species_model_6') # Index 5
ci_data_name.append(data_name+'/serial_x3_species_model_7') # Index 6

# We now use the ci_data_name list to load the image on each job, noting that in this example I'm assuming our data is
# on the Cosma data directory folder called 'example_cosma'.
cosma_data_path = '/cosma5/data/durham/'+cosma_username+'/autocti/data/'

# We then create the path to the data itself inside this folder, based on the data name and model.
data_path = cosma_data_path + ci_data_name[batch_id]+'/'

ci_datas = []

for image_index in range(len(normalizations)):

    ci_datas.append(ci_data.load_ci_data_from_fits(
                 frame_geometry=frame_geometry, ci_pattern=patterns[image_index],
                 image_path=data_path + '/ci_image_' + str(image_index) + '.fits',
                 ci_pre_cti_path=data_path+'/ci_pre_cti_'+str(image_index)+'.fits',
                 noise_map_from_single_value=4.0))

serial_cti_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                               charge_injection_mode=False, readout_offset=0)
cti_settings = arctic_settings.ArcticSettings(serial=serial_cti_settings)

# Running a pipeline is exactly the same as we're used to. We import it, make it, and run it, noting that we can
# use the lens_name's to ensure each job outputs its results to a different directory.

from workspace_jam.pipelines import serial_x3_species

pipeline = serial_x3_species.make_pipeline(pipeline_path=ci_data_name[batch_id] + '/')

pipeline.run(ci_datas=ci_datas, cti_settings=cti_settings, pool=Pool(processes=8))

# Finally, its worth us going through a batch script in detail, line by line, as you may we need to change different
# parts of this script to use different runners. Therefore, checkout the 'doc' file in the batch folder.