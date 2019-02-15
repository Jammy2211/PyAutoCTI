import logging

from autofit.tools.phase import ResultsCollection

logger = logging.getLogger(__name__)


class Pipeline(object):

    def __init__(self, *phases):
        """

        Parameters
        ----------
        phases: [ph.Phase]
            Phases
        """
        self.phases = phases

    def run(self, ci_datas, cti_settings, pool=None):
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            results.append(phase.run(ci_datas=ci_datas, cti_settings=cti_settings,
                                     previous_results=ResultsCollection(results), pool=pool))
        return results

    def __add__(self, other):
        """
        Compose two Pipelines

        Parameters
        ----------
        other: Pipeline
            Another pipeline

        Returns
        -------
        composed_pipeline: Pipeline
            A pipeline that runs all the  phases from this pipeline and then all the phases from the other pipeline
        """
        return Pipeline(*(self.phases + other.phases))
