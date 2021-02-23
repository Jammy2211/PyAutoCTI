import logging

from autofit.tools import pipeline as pl

logger = logging.getLogger(__name__)


class Pipeline(pl.Pipeline):
    def run(self, datasets, clocker, info=None, pool=None):
        def runner(phase, results):
            return phase.run(
                dataset_list=datasets,
                results=results,
                clocker=clocker,
                info=info,
                pool=pool,
            )

        return self.run_function(runner)
