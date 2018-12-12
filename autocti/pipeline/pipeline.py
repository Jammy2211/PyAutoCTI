import logging

from autofit.core.phase import ResultsCollection

from autocti.pipeline import phase as ph

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
            if not isinstance(phase, ph.HyperOnly):
                results.append(phase.run(ci_datas=ci_datas, cti_settings=cti_settings,
                                         previous_results=ResultsCollection(results), pool=pool))
            elif isinstance(phase, ph.HyperOnly):
                results[-1].hyper = phase.run(ci_datas=ci_datas, cti_settings=cti_settings,
                                              previous_results=ResultsCollection(results), pool=pool)
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
