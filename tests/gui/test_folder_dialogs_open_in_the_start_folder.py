"""The folder dialogs open in the start folder of the preferences, or in the home folder when none is set"""
import os

import pytest

from ClearMap.gui import dialog_helpers


class FakeFileDialog:
    """Records where the dialog was asked to open"""
    DontUseNativeDialog = 'DontUseNativeDialog'
    Options = staticmethod(lambda option: option)  # PyQt5 enum flags, as ClearMap 3.1 uses them
    opened_in = []

    def getExistingDirectory(self, *, parent, caption, directory, options):
        FakeFileDialog.opened_in.append(directory)
        return ''

    def close(self):
        pass


@pytest.fixture
def opened_in(monkeypatch):
    monkeypatch.setattr(dialog_helpers, 'QFileDialog', FakeFileDialog)
    FakeFileDialog.opened_in = []
    return FakeFileDialog.opened_in


@pytest.mark.parametrize('start_folder', [None, ''])  # None: not set. '': not set, as ClearMap 3.1.x wrote it
def test_without_a_start_folder_the_dialog_opens_in_the_home_folder(opened_in, start_folder):
    dialog_helpers.get_directory_dlg(start_folder)

    assert opened_in == [os.path.expanduser('~')]  # Was 'None' (a folder named None) or '.' (where ClearMap started)


def test_with_a_start_folder_the_dialog_opens_in_it(opened_in):
    dialog_helpers.get_directory_dlg('~/data')

    assert opened_in == [os.path.expanduser('~/data')]
