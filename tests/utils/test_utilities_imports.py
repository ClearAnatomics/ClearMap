"""
install_gui.sh imports ClearMap modules in a temporary environment where numpy and psutil are not installed yet: what
those modules import at the top level has to be in the standard library or in the packages of that environment.
"""


def test_utilities_imports_in_the_bootstrap_environment(import_in_bootstrap_environment):
    result = import_in_bootstrap_environment('ClearMap.Utils.utilities')

    assert result.returncode == 0, result.stderr
