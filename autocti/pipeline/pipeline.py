import logging

from autofit.tools import pipeline

logger = logging.getLogger(__name__)


class Pipeline(pipeline.Pipeline):
    def run(self, ci_datas, cti_settings, pool=None):
        def runner(phase, results):
            return phase.run(ci_datas=ci_datas, cti_settings=cti_settings, results=results, pool=pool)

        return self.run_function(runner)
