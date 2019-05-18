import logging

from autofit.tools import pipeline

logger = logging.getLogger(__name__)


class Pipeline(pipeline.Pipeline):
    def run(self, ci_datas, cti_settings, pool=None, data_name=None, assert_optimizer_pickle_matches=False):
        def runner(phase, results):
            return phase.run(ci_datas=ci_datas, cti_settings=cti_settings, results=results, pool=pool)

        return self.run_function(runner, data_name, assert_optimizer_pickle_matches=assert_optimizer_pickle_matches)
