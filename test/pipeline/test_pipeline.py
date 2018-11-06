from autocti.autofit import model_mapper as mm
from autocti.autofit import non_linear as nl
from autocti.pipeline import pipeline as pl

class DummyPhase(object):
    def __init__(self):
        self.masked_image = None
        self.previous_results = None
        self.phase_name = "dummy_phase"

    def run(self, ci_datas, cti_settings, previous_results, pool):
        self.ci_datas = ci_datas
        self.cti_settings = cti_settings
        self.previous_results = previous_results
        self.pool = pool
        return nl.Result(mm.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        pipeline = pl.Pipeline(phase_1, phase_2)

        pipeline.run(ci_datas=None, cti_settings=None, pool=None)

        assert len(phase_1.previous_results) == 0
        assert len(phase_2.previous_results) == 1

    def test_addition(self):
        phase_1 = DummyPhase()
        phase_2 = DummyPhase()
        phase_3 = DummyPhase()

        pipeline1 = pl.Pipeline(phase_1, phase_2)
        pipeline2 = pl.Pipeline(phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
