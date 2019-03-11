import logging

from autofit.tools.pipeline import Pipeline
from autofit.tools.phase import ResultsCollection

logger = logging.getLogger(__name__)


class Pipeline(Pipeline):

    def run(self, ci_datas, cti_settings, pool=None):
        results = []
        for i, phase in enumerate(self.phases):
            logger.info("Running Phase {} (Number {})".format(phase.phase_name, i))
            results.append(phase.run(ci_datas=ci_datas, cti_settings=cti_settings,
                                     previous_results=ResultsCollection(results), pool=pool))
        return results