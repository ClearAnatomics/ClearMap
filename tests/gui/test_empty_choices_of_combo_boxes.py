"""
The partner of a channel (align_with) has three states, each an item of its combo box: not chosen yet
(ALIGN_WITH_UNDECIDED, shown as is), explicitly not registered (None in the config, the item 'None'), or a partner.
A channel without layout channel is None in the config, the item 'undefined'.
The combo boxes and the config must translate one into the other (an item the box does not have leaves the previous
channel shown).
"""
import re
from pathlib import Path

import pytest

import ClearMap.gui
from ClearMap.config.registration_channels import ALIGN_WITH_UNDECIDED, ATLAS
from ClearMap.gui.params import (NO_PARTNER_TEXT, UNDEFINED_LAYOUT_CHANNEL_TEXT, align_with_choices,
                                 align_with_from_ui, align_with_to_ui, layout_channel_from_ui, layout_channel_to_ui)


def test_the_choices_of_the_partner_are_texts_for_each_state():
    assert align_with_choices(['fos', 'auto']) == [ALIGN_WITH_UNDECIDED, NO_PARTNER_TEXT, ATLAS, 'fos', 'auto']
    assert all(isinstance(choice, str) and choice for choice in align_with_choices([]))  # No empty item


def test_no_partner_is_shown_as_the_no_partner_item():
    assert align_with_to_ui(None) == NO_PARTNER_TEXT


def test_the_no_partner_item_is_none_in_the_config():
    assert align_with_from_ui(NO_PARTNER_TEXT) is None
    assert align_with_from_ui(None) is None  # What the combo box getter gives for it


@pytest.mark.parametrize('partner', [ALIGN_WITH_UNDECIDED, ATLAS, 'auto'])
def test_the_other_states_go_to_the_ui_and_back_unchanged(partner):
    assert align_with_to_ui(partner) == partner
    assert align_with_from_ui(partner) == partner


def test_no_layout_channel_is_shown_as_the_undefined_item():
    assert layout_channel_to_ui(None) == UNDEFINED_LAYOUT_CHANNEL_TEXT


def test_the_undefined_item_is_no_layout_channel_in_the_config():
    assert layout_channel_from_ui(UNDEFINED_LAYOUT_CHANNEL_TEXT) is None


def test_a_layout_channel_goes_to_the_ui_and_back_unchanged():
    assert layout_channel_to_ui('fos') == 'fos'
    assert layout_channel_from_ui('fos') == 'fos'


def test_the_no_partner_item_of_the_designer_file_is_the_one_of_the_code():
    ui_file = Path(ClearMap.gui.__file__).parent / 'creator' / 'registration_params.ui'
    box = re.search(r'<widget class="QComboBox" name="alignWithComboBox">(.*?)</widget>', ui_file.read_text(),
                    re.DOTALL).group(1)
    assert NO_PARTNER_TEXT in re.findall(r'<string>(.*?)</string>', box)
