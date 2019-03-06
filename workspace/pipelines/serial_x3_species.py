from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autofit.mapper import prior_model
from autocti.charge_injection import ci_hyper
from autocti.pipeline import pipeline as pl
from autocti.pipeline import phase as ph
from autocti.model import arctic_params

# In this pipeline, we'll perform an analysis which fits three serial trap species to a set of charge
# injection imaging data. This will include a hyper-phase which scales the noise in the analysis, to prevent
# over-fitting the highest S/N charge injection images. The pipeline uses four phases:

# Phase 1) Fit a small section (10 rows of every charge injection) using a serial CTI model
#          with 1 trap species and a model for the serial CCD volume filling parameters.

# Phase 2) Fit a small section (again, 10 rows) using a serial CTI model with 3 trap species and a model for the serial
#          CCD volume filling parameters. The priors on trap densities and volume filling parameters are initialized
#          from the results of phase 1.

# Phase 3) Use the best-fit model from phase 2 to scale the noise of each image, to ensure that the higher and
#          lower S/N images are weighted more equally in their contribution to the likelihood.

# Phase 4) Refit the phase 2 model, using priors initialized from the results of phase 2 and the scaled noise-map
#          computed in phase 3.

def make_pipeline(pipeline_path=''):

    pipeline_name = 'pipeline_serial_x3_species'
    pipeline_path = pipeline_path + pipeline_name

    ### PHASE 1 ###

    # In phase 1, we will fit the data with a one species serial CTI model and serial CCD filling model. In this
    # phase we will:

    # 1) Extract and fit the 10 rows of each charge injection region which is furthest from the clocking direction (and
    # therefore least affected by parallel CTI).

    class SerialPhase(ph.SerialPhase):

        def pass_priors(self, previous_results):
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase1 = SerialPhase(optimizer_class=nl.MultiNest,
                           serial_species=[prior_model.PriorModel(arctic_params.Species)],
                           serial_ccd=arctic_params.CCD, rows=(0,10),
                           phase_name=pipeline_path + '/phase_1_x1_species')

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the data with a three species serial CTI model and serial CCD filling model. In this
    # phase we will:

    # 1) As in phase 1, extract and fit the 10 rows of charge injection imaging data closest to the read-out
    # register.
    # 2) Use priors on the trap density and ccd volume filling parameters based on the results of phase 1.

    class SerialPhase(ph.SerialPhase):

        def pass_priors(self, previous_results):

            previous_total_density = previous_results[-1].constant.serial_species[0].trap_density

            self.serial_species[0].trap_density = prior.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)
            self.serial_species[1].trap_density = prior.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)
            self.serial_species[2].trap_density = prior.UniformPrior(lower_limit=0.0, upper_limit=previous_total_density)
            self.serial_species[0].trap_lifetime = prior.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.serial_species[1].trap_lifetime = prior.UniformPrior(lower_limit=0.0, upper_limit=30.0)
            self.serial_species[2].trap_lifetime = prior.UniformPrior(lower_limit=0.0, upper_limit=30.0)

            self.serial_ccd.well_notch_depth = previous_results[0].variable.serial_ccd.well_notch_depth
            self.serial_ccd.well_fill_beta = previous_results[0].variable.serial_ccd.well_fill_beta
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase2 = SerialPhase(serial_species=[prior_model.PriorModel(arctic_params.Species),
                                         prior_model.PriorModel(arctic_params.Species),
                                         prior_model.PriorModel(arctic_params.Species)],
                         serial_ccd=arctic_params.CCD, rows=(0,10), optimizer_class=nl.MultiNest,
                         phase_name=pipeline_path + '/phase_2_x3_species_initialize')

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.2

    ### PHASE 3 ###

    # The best fit model of phase 2 is used to create a 'noise-scaling' map for every charge injection image. These
    # noise-scaling maps are used in conjunction with 'hyper-noise' models to scale the noise-maps in a way that
    # equally weights the fit across all charge injection images.

    class SerialHyperModelFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, previous_results):

            self.serial_species = previous_results[1].constant.serial_species
            self.serial_ccd = previous_results[1].constant.serial_ccd

    phase3 = SerialHyperModelFixedPhase(serial_species=[prior_model.PriorModel(arctic_params.Species),
                                                        prior_model.PriorModel(arctic_params.Species),
                                                        prior_model.PriorModel(arctic_params.Species)],
                                        serial_ccd=arctic_params.CCD, rows=None,
                                        optimizer_class=nl.MultiNest,
                                        phase_name=pipeline_path + '/phase_3_noise_scaling')

    ### PHASE 4 ###

    # In phase 4, we will fit the data with a 3 species serial CTI model and serial CCD filling model. In this
    # phase we will:

    # 1) Use the complete charge injection image, as opposed to extracting a sub-set of rows.
    # 2) Use the scaled noise-map computed in phase 3.
    # 3) Initialize the priors on the serial CTI model from the results of phase 2.

    class SerialHyperFixedPhase(ph.SerialHyperPhase):

        def pass_priors(self, previous_results):

            self.hyper_noise_scalars = previous_results[2].constant.hyper_noise_scalars
            self.serial_species = previous_results[1].variable.serial_species
            self.serial_ccd = previous_results[1].variable.serial_ccd
            self.serial_ccd.well_fill_alpha = 1.0
            self.serial_ccd.well_fill_gamma = 0.0

    phase4 = SerialHyperFixedPhase(serial_species=[prior_model.PriorModel(arctic_params.Species),
                                                   prior_model.PriorModel(arctic_params.Species),
                                                   prior_model.PriorModel(arctic_params.Species)],
                                   serial_ccd=arctic_params.CCD, rows=None,
                                   optimizer_class=nl.MultiNest,
                                   phase_name=pipeline_path + '/phase_4_final')

    # For the final CTI model, constant efficiency mode has a tendency to sample parameter space too fast and infer an
    # inaccurate model. Thus, we turn it off for phase 2.

    phase4.optimizer.const_efficiency_mode = False
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.3

    return pl.Pipeline(phase1, phase2, phase3, phase4)