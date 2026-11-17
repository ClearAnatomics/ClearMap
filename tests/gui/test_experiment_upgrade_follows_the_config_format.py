"""
The GUI decides whether to upgrade an experiment with ``config_needs_upgrade`` and converts it to
``CURRENT_CONFIG_VERSION``, never by comparing it with the version of the package.

ClearMap 3.1.1 compared the "3.1.0" of every config with its own "3.1.1": it offered to upgrade every experiment and
the upgrade failed. ``ClearMap.gui.app`` builds the application when it is imported, so its source is read instead.
"""
import ast
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parents[2] / 'ClearMap' / 'gui' / 'app.py'
PACKAGE_VERSION_NAMES = ('CLEARMAP_VERSION', '__version__', 'applicationVersion', '_version')
CONVERTING_CALLS = ('upgrade_configs', 'convert_versions')


def _names_in(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            yield sub.id
        elif isinstance(sub, ast.Attribute):
            yield sub.attr


def _uses_the_package_version(node):
    return any(name in PACKAGE_VERSION_NAMES for name in _names_in(node))


def _called_name(call):
    function = call.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, 'id', None)


def test_no_comparison_in_the_gui_app_involves_the_package_version():
    tree = ast.parse(APP_SOURCE.read_text(encoding='utf-8'))
    comparisons = [f'line {node.lineno}' for node in ast.walk(tree)
                   if isinstance(node, ast.Compare) and _uses_the_package_version(node)]

    assert not comparisons, f'app.py compares with the package version at {comparisons}; use config_needs_upgrade'


def test_the_gui_app_converts_experiments_to_the_config_format():
    tree = ast.parse(APP_SOURCE.read_text(encoding='utf-8'))
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and _called_name(node) in CONVERTING_CALLS]
    to_the_package_version = [f'line {call.lineno}' for call in calls
                              if any(_uses_the_package_version(argument) for argument in call.args)]

    assert calls, 'app.py no longer converts experiments: update this test'
    assert not to_the_package_version, f'app.py converts to the package version at {to_the_package_version}'
