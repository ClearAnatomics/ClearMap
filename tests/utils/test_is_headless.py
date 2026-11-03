"""is_headless tells early_boot (and the GUI) not to prompt: CLEARMAP_HEADLESS, or Linux without a display"""
import pytest

from ClearMap.Utils import utilities


@pytest.fixture
def linux_without_display(monkeypatch):
    monkeypatch.setattr(utilities.platform, 'system', lambda: 'Linux')
    for name in ('CLEARMAP_HEADLESS', 'DISPLAY', 'QT_QPA_PLATFORM'):
        monkeypatch.delenv(name, raising=False)


def test_linux_without_display_is_headless(linux_without_display):
    assert utilities.is_headless()


def test_linux_with_a_display_is_not_headless(linux_without_display, monkeypatch):
    monkeypatch.setenv('DISPLAY', ':0')

    assert not utilities.is_headless()


def test_the_environment_variable_makes_a_machine_with_a_display_headless(linux_without_display, monkeypatch):
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setenv('CLEARMAP_HEADLESS', '1')

    assert utilities.is_headless()


def test_other_platforms_are_not_headless_without_the_variable(linux_without_display, monkeypatch):
    monkeypatch.setattr(utilities.platform, 'system', lambda: 'Darwin')

    assert not utilities.is_headless()
