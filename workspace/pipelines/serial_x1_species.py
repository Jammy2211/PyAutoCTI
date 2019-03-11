from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autofit.mapper import prior_model
from autocti.pipeline import pipeline as pl
from autocti.pipeline import phase as ph
from autocti.model import arctic_params

# In this pipeline, we'll perform a basic analysis which fits a single serial trap species to a set of charge
# injection imaging data. This will include a hyper-phase which scales the noise in the analysis, to prevent
# over-fitting the highest S/N charge injection images. The pipeline uses three phases:

# Phase 1) Fit a small section (the top 10 rows of every charge injection) using a serial CTI model
#          with 1 trap species and a model for the serial CCD volume filling parameters.

# Phase 2) Use the best-fit model from phase 1 to scale the noise of each image, to ensure that the higher and
#          lower S/N images are weighted more equally in their contribution to the likelihood.

# Phase 3) Refit the phase 1 model, using priors initialized from the results of phase 1 and the scaled noise-map
#          computed in phase 2.

def make_pipeline(phase_folders=None):

    pipeline_name = 'pipeline_serial_x1_species'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will fit the data with a one species serial CTI model and serial CCD filling model. In this
    # phase we will:

    # 1) Extract and fit the 60 columns of charge injection imaging data closest to the read-out register (and
    # therefore least affected by serial CTI).

    class SerialPhase(ph.SerialPhase):

        def pass_priors(self, results):
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = SerialPhase(phase_name='phase_1_initialize', phase_folders=phase_folders,
                         optimizer_class=nl.MultiNest,
                         serial_species=[prior_model.PriorModel(arctic_params.Species)],
                         serial_ccd=arctic_params.CCD, rows=(0,10))

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # The best fit model of phase 1 is used to create a 'noise-scaling' map for every charge injection image. These
    # noise-scaling maps are used in conjuction with 'hyper-noise' models to scale the noise-maps in a way that
    # equally weights the fit across all charge injection images.

    # 1) Extract and fit the 60 columns of charge injection imaging data closest to the read-out register (and
    # therefore least affected by serial CTI).

    class SerialHyperModelFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, results):
            self.serial_species = results.from_phase('phase_1_initialize').constant.serial_species
            self.serial_ccd = results.from_phase('phase_1_initialize').constant.serial_ccd

    phase2 = SerialHyperModelFixedPhase(phase_name='phase_2_noise_scaling', phase_folders=phase_folders,
                                        serial_species=[prior_model.PriorModel(arctic_params.Species)],
                                        serial_ccd=arctic_params.CCD,
                                        optimizer_class=nl.MultiNest, rows=(0, 10))

    ### PHASE 2 ###

    # In phase 2, we will fit the data with a one species serial CTI model and serial CCD filling model. In this
    # phase we will:

    # 1) Use the complete charge injection image, as opposed to extracting a sub-set of rows.
    # 2) Use the scaled noise-map computed in phase 2.
    # 3) Initialize the priors on the serial CTI model from the results of phase 1.

    class SerialHyperFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, results):

            self.hyper_noise_scalars = results.from_phase('phase_2_noise_scaling').constant.hyper_noise_scalars
            self.serial_species = results.from_phase('phase_1_initialize').variable.serial_species
            self.serial_ccd = results.from_phase('phase_1_initialize').variable.serial_ccd
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase3 = SerialHyperFixedPhase(phase_name='phase_3_final', phase_folders=phase_folders,
                                   optimizer_class=nl.MultiNest)

    # For the final CTI model, constant efficiency mode has a tendancy to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pl.Pipeline(pipeline_name, phase1, phase2, phase3)