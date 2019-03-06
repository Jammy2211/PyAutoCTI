from autocti.data import cti_image
from autocti.charge_injection import ci_frame

from autocti.model import arctic_settings
from autocti.model import arctic_params

import os
import numpy as np

# This tool allows one to make simulated images of parallel charge transfer inefficiency, typically using simple image
# configurations to highlight the effects of capture and trailing.

# The 'data name' is the name of the lens in the data folder, e.g:

# The image will be output as '/workspace/data/data_name/image.fits'.
# The pre cti image will be output as '/workspace/data/data_name/image_pre_cti.fits'.

data_name = 'parallel_image_horizontal_line'

shape = (5, 5) # The shape determines the shape of the image simulated.

# Lets use this shape to create a NumPy array of zeros of this shape, and add a horiztonal line acrosss the image
# perpedicular to the direction of read-out and thus CTI.
image = np.zeros(shape=shape)
image[1,0:5] = 1000.0

# The frame geometry of the image being simuated.
frame_geometry = ci_frame.QuadGeometryEuclid.bottom_left()
# frame_geometry = ci_frame.QuadGeometryEuclid.bottom_right()
# frame_geometry = ci_frame.QuadGeometryEuclid.top_left()
# frame_geometry = ci_frame.QuadGeometryEuclid.top_right()


# Set this image up as a CTI image, so that we can add CTI to it.
image_pre_cti = cti_image.ImageFrame(frame_geometry=frame_geometry, array=image)

# The CTI settings of arCTIc, which models the CCD read-out including CTI.
parallel_cti_settings = arctic_settings.Settings(well_depth=84700, niter=1, express=2, n_levels=2000,
                                                 charge_injection_mode=False, readout_offset=0)
cti_settings = arctic_settings.ArcticSettings(parallel=parallel_cti_settings)

# The CTI model parameters of arCTIc, which includes each trap species density / lifetime and the CCD properties for
# parallel charge transfer.
parallel_species = arctic_params.Species(trap_density=0.5, trap_lifetime=4.0)
parallel_ccd = arctic_params.CCD(well_notch_depth=1.0e-4, well_fill_beta=0.58, well_fill_gamma=0.0, well_fill_alpha=1.0)
cti_params = arctic_params.ArcticParams(parallel_species=[parallel_species], parallel_ccd=parallel_ccd)

# Use the pre-cti image to create a post-cti image, by passing it the arctic setting and CTI model above.
image_post_cti = image_pre_cti.add_cti_to_image(cti_params=cti_params, cti_settings=cti_settings)

# Setup the post cti image as an image frame (this should be inbuilt into array_finalize functions in the future)
image_post_cti = cti_image.ImageFrame(frame_geometry=frame_geometry, array=image_post_cti)

# Now, lets output this simulated ccd-data to the data folder. First, we'll get a relative path.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Check a folder of the data_name exists in the data folder for the images to be output. If it doesn't make it.
if not os.path.exists(path+'/data/'+data_name):
    os.makedirs(path+'/data/'+data_name)

# Now, output every image to the data folder.
image_pre_cti.output_as_fits(file_path=path+'/data/'+data_name+'/image_pre_cti.fits', overwrite=True)
image_post_cti.output_as_fits(file_path=path+'/data/'+data_name+'/image_post_cti.fits', overwrite=True)