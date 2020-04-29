import logging

import autofit as af

logger = logging.getLogger(__name__)


class Pipeline(af.Pipeline):
    def run(self, datasets, clocker, info=None, pool=None):
        def runner(phase, results):
            return phase.run(datasets=datasets, results=results, info=info, pool=pool)

        return self.run_function(runner)
