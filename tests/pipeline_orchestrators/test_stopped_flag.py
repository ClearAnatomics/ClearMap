"""
The Abort button calls ``stop_process``, which sets ``stopped``: the steps guarded by ``check_stopped`` then return
at once. Nothing but ``reset_stopped`` clears the flag, so a new run has to call it.
"""
from unittest.mock import MagicMock

from ClearMap.Utils.utilities import check_stopped
from ClearMap.pipeline_orchestrators.generic_orchestrators import PipelineOrchestrator


class GuardedOrchestrator(PipelineOrchestrator, abstract=True):
    # abstract: it belongs to no pipeline, so the orchestrator contract tests do not list it

    def __init__(self):
        super().__init__(MagicMock())
        self.n_runs = 0

    @check_stopped
    def step(self):
        self.n_runs += 1


def test_a_stopped_orchestrator_skips_its_guarded_steps():
    orchestrator = GuardedOrchestrator()
    orchestrator.stop_process()
    orchestrator.step()
    assert orchestrator.stopped
    assert orchestrator.n_runs == 0


def test_a_guarded_step_runs_again_after_the_stop_is_reset():
    orchestrator = GuardedOrchestrator()
    orchestrator.stop_process()
    orchestrator.reset_stopped()
    orchestrator.step()
    assert not orchestrator.stopped
    assert orchestrator.n_runs == 1
