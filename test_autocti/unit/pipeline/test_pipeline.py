import builtins
from autofit.mapper import model
from autofit.non_linear import abstract_search
from autofit.tools import phase
from autocti.pipeline import pipeline as pl
import pytest


class MockFile(object):
    def __init__(self):
        self.text = None
        self.filename = None

    def write(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture(name="mock_files", autouse=True)
def make_mock_file(monkeypatch):
    files = []

    def mock_open(filename, flag, *args, **kwargs):
        assert flag in ("w+", "w+b", "a")
        file = MockFile()
        file.filename = filename
        files.append(file)
        return file

    monkeypatch.setattr(builtins, "open", mock_open)
    return files


class Optimizer(object):
    def __init__(self, phase_name="dummy_phase"):
        self.phase_name = phase_name
        self.phase_path = ""


class DummyPhase(phase.AbstractPhase):
    def make_result(self, result, analysis):
        pass

    def __init__(self, phase_name, phase_tag=None):
        super().__init__(phase_name)
        self.masked_image = None
        self.results = None
        self.phase_name = phase_name
        self.phase_tag = phase_tag
        self.phase_path = phase_name

        self.search = Optimizer(phase_name)

    def run(self, datasets, clocker, results, pool):
        self.datasets = datasets
        self.clocker = clocker
        self.results = results
        self.pool = pool
        return abstract_search.Result(model.ModelInstance(), 1)


class TestPipeline(object):
    def test_run_pipeline(self):
        phase1 = DummyPhase(phase_name="dummy1")
        phase2 = DummyPhase(phase_name="dummy2")
        pipeline = pl.Pipeline("", phase_1, phase_2)

        pipeline.run(datasets=None, clocker=None, pool=None)

        assert len(phase1.results) == 2
        assert len(phase2.results) == 2

    def test_addition(self):
        phase1 = DummyPhase(phase_name="dummy1")
        phase2 = DummyPhase(phase_name="dummy2")
        phase3 = DummyPhase(phase_name="dumy3")

        pipeline1 = pl.Pipeline("", phase_1, phase_2)
        pipeline2 = pl.Pipeline("", phase_3)

        assert (phase_1, phase_2, phase_3) == (pipeline1 + pipeline2).phases
