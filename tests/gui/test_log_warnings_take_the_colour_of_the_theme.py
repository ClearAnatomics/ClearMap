"""
The log writes its warnings (and the disclaimer under a warning traceback) in the warning colour that the window gives
it, the one of the theme (gui.style.message_colours): yellow is unreadable on a light background.
"""
from ClearMap.gui import style
from ClearMap.gui.gui_logging import Printer

WARNING_LINE = '/a/file.py:12: UserWarning: something'


def make_printer(**kwargs):
    printer = Printer(**kwargs)
    printer.written = []
    printer.write = printer.written.append
    return printer


def test_the_log_writes_warnings_in_the_warning_colour_it_is_given():
    printer = make_printer(color='red', warning_colour=style.MESSAGE_COLOURS_ON_LIGHT.warning)

    assert printer.colourise(WARNING_LINE) == (f'<p style="color:{style.MESSAGE_COLOURS_ON_LIGHT.warning}">'
                                               f'{WARNING_LINE}</p>')


def test_the_warning_disclaimer_is_in_the_warning_colour_it_is_given():
    printer = make_printer(warning_colour=style.MESSAGE_COLOURS_ON_LIGHT.warning)

    printer.warning_disclaimer()

    assert f'color:{style.MESSAGE_COLOURS_ON_LIGHT.warning}' in printer.written[0]


def test_by_default_the_log_warnings_are_in_the_yellow_of_the_dark_theme():
    printer = make_printer(color='red')

    assert f'color:{style.MESSAGE_COLOURS_ON_DARK.warning}' in printer.colourise(WARNING_LINE)
