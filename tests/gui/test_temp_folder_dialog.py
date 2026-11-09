"""
Without a valid temporary folder in the environment or the machine config, the early boot asks the user: a folder
dialog, then a text entry. The dialogs live in ClearMap.gui, so that early_boot imports no Qt. Qt is faked here: the
sandbox and the CI have no display.
"""
from types import SimpleNamespace

import pytest

from ClearMap.config.early_boot import MachineConfig
from ClearMap.gui import early_boot_dialogs


class FakeFileDialog:
    ShowDirsOnly, DontResolveSymlinks = 1, 16
    chosen = ''

    @classmethod
    def getExistingDirectory(cls, parent, caption, directory, options):
        return cls.chosen


class FakeInputDialog:
    typed = ('', False)

    @classmethod
    def getText(cls, parent, title, label):
        return cls.typed


class FakeMessageBox:
    shown = []

    @classmethod
    def warning(cls, parent, title, text):
        cls.shown.append(title)

    @classmethod
    def critical(cls, parent, title, text):
        cls.shown.append(title)


def no_application():
    return None


def any_folder_is_suitable(path):
    return True


@pytest.fixture()
def dialogs(monkeypatch):
    monkeypatch.setattr(early_boot_dialogs, 'ensure_qapp', no_application)
    for name, fake in (('QFileDialog', FakeFileDialog), ('QInputDialog', FakeInputDialog),
                       ('QMessageBox', FakeMessageBox)):
        monkeypatch.setattr(early_boot_dialogs, name, fake)
    monkeypatch.setattr(FakeFileDialog, 'chosen', '')
    monkeypatch.setattr(FakeInputDialog, 'typed', ('', False))
    monkeypatch.setattr(FakeMessageBox, 'shown', [])
    return SimpleNamespace(file=FakeFileDialog, input=FakeInputDialog, messages=FakeMessageBox)


def test_a_suitable_folder_chosen_in_the_dialog_is_the_temporary_folder(dialogs, tmp_path):
    dialogs.file.chosen = str(tmp_path)

    assert early_boot_dialogs.prompt_tmp_folder(min_free_gb=5, is_valid=any_folder_is_suitable) == str(tmp_path)
    assert dialogs.messages.shown == []


def test_an_unsuitable_folder_is_refused_and_the_typed_one_is_taken(dialogs, tmp_path):
    unsuitable, typed = tmp_path / 'small', tmp_path / 'large'
    dialogs.file.chosen = str(unsuitable)
    dialogs.input.typed = (f' {typed} ', True)

    folder = early_boot_dialogs.prompt_tmp_folder(min_free_gb=5, is_valid=frozenset([typed]).__contains__)

    assert folder == str(typed)
    assert dialogs.messages.shown == ['Folder not suitable']


def test_no_folder_given_is_none(dialogs):
    assert early_boot_dialogs.prompt_tmp_folder(min_free_gb=5, is_valid=any_folder_is_suitable) is None


def test_the_machine_config_asks_with_its_free_space_and_its_validity_rule(monkeypatch):
    asked = {}

    def answer(**kwargs):
        asked.update(kwargs)
        return '/data/tmp'
    monkeypatch.setattr(early_boot_dialogs, 'prompt_tmp_folder', answer)
    machine_config = MachineConfig()

    assert machine_config.prompt_tmp_folder() == '/data/tmp'
    assert asked == {'min_free_gb': machine_config.min_temp_folder_gb, 'is_valid': machine_config._is_valid_tmp}
