"""
An orchestrator reports its progress whether a display was set or not: until one is, its watcher is a
ProgressWatcherBase, which shows nothing. Its methods used to test for None at each call, and a watcher left unset in
a headless run was only safe as long as every call was guarded.
"""
from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.events import ProgressSubStepNameChanged
from ClearMap.Utils.progress import ProgressWatcher, ProgressWatcherBase
from ClearMap.pipeline_orchestrators.generic_orchestrators import GroupOrchestratorBase, PipelineOrchestrator


class FakeCoordinator:
    def __init__(self):
        self._bus = EventBus()


class Orchestrator(PipelineOrchestrator, abstract=True):
    """The progress methods of every orchestrator (abstract: no pipeline to declare)"""


class FakeGroupController:
    pass


class SubStepNames:
    def __init__(self, bus):
        self.names = []
        bus.subscribe(ProgressSubStepNameChanged, self.record)

    def record(self, event):
        self.names.append(event.name)


def report_some_progress(orchestrator):
    orchestrator.set_watcher_step('Cell detection')
    orchestrator.prepare_watcher_for_substep(10, None, 'Blocks', increment_main=True)
    orchestrator.update_watcher_progress(3)
    orchestrator.update_watcher_main_progress()


def test_an_orchestrator_reports_its_progress_without_a_display():
    orchestrator = Orchestrator(FakeCoordinator())
    assert type(orchestrator.progress_watcher) is ProgressWatcherBase
    report_some_progress(orchestrator)


def test_the_progress_goes_to_the_watcher_set():
    orchestrator = Orchestrator(FakeCoordinator())
    bus = EventBus()
    sub_steps = SubStepNames(bus)
    orchestrator.set_progress_watcher(ProgressWatcher(bus))
    report_some_progress(orchestrator)
    assert sub_steps.names == ['Blocks']


def test_a_group_orchestrator_reports_its_progress_without_a_display():
    GroupOrchestratorBase(group_controller=FakeGroupController())._increment_progress_main(1)
