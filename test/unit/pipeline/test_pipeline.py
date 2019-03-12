from autofit.mapper import model_mapper as mm
from autofit.optimize import non_linear as nl

from autocti.pipeline import pipeline as pl


class DummyPhase(object):
    def __init__(self, phase_name):
        self.masked_image = None
        self.results = None
        self.phase_name = phase_name

        class Optimizer(object):
            @property
            def phase_name(self):
                return phase_name

        self.optimizer = Optimizer()

    def run(self, ci_datas, cti_settings, results, pool):
        self.ci_datas = ci_datas
        self.cti_settings = cti_settings
        self.results = results
        self.pool = pool
        return nl.Result(mm.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase(phase_name='dummy1')
        phase_2 = DummyPhase(phase_name='dummy2')
        pipeline = pl.Pipeline('', phase_1, phase_2)

        pipeline.run(ci_datas=None, cti_settings=None, pool=None)

        assert len(phase_1.results) == 2
        assert len(phase_2.results) == 2

    def test_addition(self):
        phase_1 = DummyPhase(phase_name='dummy1')
        phase_2 = DummyPhase(phase_name='dummy2')
        phase_3 = DummyPhase(phase_name='dumy3')

        pipeline1 = pl.Pipeline('', phase_1, phase_2)
        pipeline2 = pl.Pipeline('', phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
