from autocti.mask import mask as msk
from autocti.pipeline.phase.settings import SettingsPhaseCIImaging
from autocti.pipeline.phase import abstract
from autocti.pipeline.phase import extensions
from autocti.pipeline.phase.dataset.result import Result
from autofit.non_linear.paths import convert_paths
from autofit.tools.phase import Dataset
from autofit.tools.phase_property import PhaseProperty
from autofit.tools.pipeline import ResultsCollection


class PhaseDataset(abstract.AbstractPhase):

    parallel_traps = PhaseProperty("parallel_traps")
    serial_traps = PhaseProperty("serial_traps")
    parallel_ccd = PhaseProperty("parallel_ccd")
    serial_ccd = PhaseProperty("serial_ccd")

    Result = Result

    @convert_paths
    def __init__(
        self,
        paths,
        search,
        parallel_traps=None,
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
        settings=SettingsPhaseCIImaging(),
    ):
        """

        A phase in an lens pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        """

        paths.tag = settings.phase_tag

        super().__init__(paths=paths, search=search)

        self.parallel_traps = parallel_traps or []
        self.parallel_ccd = parallel_ccd
        self.serial_traps = serial_traps or []
        self.serial_ccd = serial_ccd
        self.settings = settings

    def run(self, datasets: Dataset, clocker, results=None, info=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask
            The default masks passed in by the pipeline
        results: autofit.tools.pipeline.ResultsCollection
            An object describing the results of the last phase or None if no phase has been executed
        dataset: scaled_array.ScaledSquarePixelArray
            An masked_imaging that has been masked

        Returns
        -------
        result: AbstractPhase.Result
            A result object comprising the best fit model and other hyper_galaxies.
        """
        #    self.save_metadata(dataset=datasets)
        #    self.save_dataset(dataset=datasets)
        #    self.save_mask(masks)

        self.model = self.model.populate(results)

        results = results or ResultsCollection()

        analysis = self.make_analysis(
            datasets=datasets, clocker=clocker, results=results, pool=pool
        )

        #    phase_attributes = self.make_phase_attributes(analysis=analysis)
        #    self.save_phase_attributes(phase_attributes=phase_attributes)

        result = self.run_analysis(analysis=analysis, info=info)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, results=None, pool=None):
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
            An lens object that the non-linear search calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def mask_for_analysis_from_cosmic_ray_map(self, cosmic_ray_map, mask):

        cosmic_ray_mask = (
            msk.Mask.from_cosmic_ray_map_buffed(
                cosmic_ray_map=cosmic_ray_map, settings=self.settings.settings_mask
            )
            if cosmic_ray_map is not None
            else None
        )

        if cosmic_ray_map is not None:
            return mask + cosmic_ray_mask

        return mask

    def extend_with_hyper_noise_phases(self):
        return extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=(extensions.HyperNoisePhase,)
        )
