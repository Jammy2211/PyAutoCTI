import autofit as af
from autofit.tools.phase import Dataset
from autocti.pipeline.phase import abstract
from autocti.pipeline.phase import extensions
from autocti.pipeline.phase.dataset.result import Result


class PhaseDataset(abstract.AbstractPhase):

    parallel_traps = af.PhaseProperty("parallel_traps")
    serial_traps = af.PhaseProperty("serial_traps")
    parallel_ccd_volume = af.PhaseProperty("parallel_ccd_volume")
    serial_ccd_volume = af.PhaseProperty("serial_ccd_volume")

    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        parallel_traps=(),
        parallel_ccd_volume=None,
        serial_traps=(),
        serial_ccd_volume=None,
        non_linear_class=af.MultiNest,
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        """

        super().__init__(paths=paths, non_linear_class=non_linear_class)

        self.parallel_traps = parallel_traps
        self.parallel_ccd_volume = parallel_ccd_volume
        self.serial_traps = serial_traps
        self.serial_ccd_volume = serial_ccd_volume

    def run(self, datasets: Dataset, masks, results=None, info=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        masks: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        datasets: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """
        #    self.save_metadata(dataset=datasets)
        #    self.save_dataset(dataset=datasets)
        #    self.save_mask(masks)
        self.save_meta_dataset(meta_dataset=self.meta_dataset)
        self.save_info(info=info)

        self.model = self.model.populate(results)

        results = results or af.ResultsCollection()

        analysis = self.make_analysis(
            datasets=datasets, cti_settings=None, results=results
        )

        #    phase_attributes = self.make_phase_attributes(analysis=analysis)
        #    self.save_phase_attributes(phase_attributes=phase_attributes)

        self.customize_priors(results)
        self.assert_and_save_pickle()

        result = self.run_analysis(analysis)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, mask, results=None, pool=None):
        """
        Create an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the non-linear optimizer calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def extend_with_hyper_noise_phases(self):
        return extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=(extensions.HyperNoisePhase,)
        )
