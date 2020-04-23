import logging

import autofit as af

logger = logging.getLogger(__name__)


class Pipeline(af.Pipeline):
    def run(self, ci_datas, clocker, pool=None, data_name=None):
        def runner(phase, results):
            return phase.run(
                ci_datas=ci_datas, clocker=clocker, results=results, pool=pool
            )

        return self.run_function(runner, data_name)
