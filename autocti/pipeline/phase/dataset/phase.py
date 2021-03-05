from autocti.mask import mask as msk
from autocti.pipeline.phase.settings import SettingsPhaseCIImaging
from autocti.pipeline.phase.abstract import phase as abstract_phase
from autocti.pipeline.phase import extensions
from autocti.pipeline.phase.dataset.result import Result
from autofit.tools.phase import Dataset
from autofit.tools.phase_property import PhaseProperty
from autofit.tools.pipeline import ResultsCollection


class PhaseDataset(abstract_phase.AbstractPhase):

    parallel_traps = PhaseProperty("parallel_traps")
    serial_traps = PhaseProperty("serial_traps")
    parallel_ccd = PhaseProperty("parallel_ccd")
    serial_ccd = PhaseProperty("serial_ccd")

    Result = Result

    def __init__(
        self,
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

        super().__init__(search=search)

        self.parallel_traps = parallel_traps or []
        self.parallel_ccd = parallel_ccd
        self.serial_traps = serial_traps or []
        self.serial_ccd = serial_ccd
        self.settings = settings

    def run(self, dataset_list: Dataset, clocker, results=None, info=None, pool=None):
        """
        Run this phase.

        Parameters
        ----------
        positions
        mask: Mask2D
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

        self.modify_search_paths()

        analysis = self.make_analysis(
            datasets=dataset_list, clocker=clocker, results=results, pool=pool
        )

        #    attributes = self.make_attributes(analysis=analysis)
        #    self.save_attributes(attributes=attributes)

        result = self.run_analysis(analysis=analysis, info=info)

        return self.make_result(result=result, analysis=analysis)

    def make_analysis(self, dataset, results=None, pool=None):
        """
        Returns an lens object. Also calls the prior passing and masked_imaging modifying functions to allow child
        classes to change the behaviour of the phase.

        Parameters
        ----------
        positions
        mask: Mask2D
            The default masks passed in by the pipeline
        dataset: im.Imaging
            An masked_imaging that has been masked
        results: autofit.tools.pipeline.ResultsCollection
            The result from the previous phase

        Returns
        -------
        lens : Analysis
            An lens object that the `NonLinearSearch` calls to determine the fit of a set of values
        """
        raise NotImplementedError()

    def mask_for_analysis_from_cosmic_ray_map(self, cosmic_ray_map, mask):

        cosmic_ray_mask = (
            msk.Mask2D.from_cosmic_ray_map_buffed(
                cosmic_ray_map=cosmic_ray_map, settings=self.settings.settings_mask
            )
            if cosmic_ray_map is not None
            else None
        )

        if cosmic_ray_map is not None:
            return mask + cosmic_ray_mask

        return mask

    def modify_search_paths(self):

        self.search.paths.tag = self.settings.phase_tag

    def extend_with_hyper_noise_phases(self):
        return extensions.CombinedHyperPhase(
            phase=self, hyper_phase_classes=[extensions.HyperNoisePhase]
        )
