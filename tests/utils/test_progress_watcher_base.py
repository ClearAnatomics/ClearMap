"""
ProgressWatcherBase is what any code may call on a progress watcher, and the watcher that shows nothing (the one the
code gets when no display was set). A member added to ProgressWatcher but not to it would only fail in the runs
without a display.
"""
import pytest

from ClearMap.Utils.event_bus import EventBus
from ClearMap.Utils.progress import ProgressWatcher, ProgressWatcherBase


def public_members(cls):
    return {name for name in dir(cls) if not name.startswith('_')}


def test_the_base_defines_every_public_member_of_the_watcher():
    watcher_members = public_members(ProgressWatcher) | public_members(ProgressWatcher(EventBus()))
    specific_to_the_watcher_that_publishes = {'bus'}
    assert watcher_members - specific_to_the_watcher_that_publishes <= public_members(ProgressWatcherBase)


def test_the_watcher_overrides_every_method_of_the_base():
    """A method left to the base in ProgressWatcher would silently do nothing in the GUI"""
    not_overridden = [name for name in public_members(ProgressWatcherBase)
                      if getattr(ProgressWatcher, name) is getattr(ProgressWatcherBase, name)
                      and name not in ('log_path', 'pattern')]
    assert not_overridden == []


@pytest.fixture
def base_watcher():
    return ProgressWatcherBase()


def test_the_base_watcher_does_nothing_whatever_it_is_told(base_watcher):
    base_watcher.setup(main_step_name='Cell map', main_step_length=3, sub_step_length=10, pattern='done')
    base_watcher.prepare_for_substep(10, pattern='done', step_name='Detection')
    base_watcher.main_step_name = 'Tube map'
    base_watcher.max_progress = 50
    base_watcher.increment(4)
    base_watcher.increment_main_progress()
    base_watcher.start_polling()
    base_watcher.stop_polling()
    base_watcher.finish()
    assert (base_watcher.get_progress(), base_watcher.max_progress, base_watcher.main_step_name) == (0, 100, 'Processing')
