"""
Qt 6.11.1 crashes on Linux (segmentation fault in FcCharSetHasChar) as soon as a widget shows a character missing from
the default font when a fallback font has no charset. qtbase 91d7ff41af58 fixed it in Qt 6.11.3, and a solve of
ClearMap3.yml without a pin chose 6.11.1, so the environment file asks for 6.11.3 or later.
"""
from pathlib import Path

import yaml
from packaging.version import Version

from ClearMap.Utils.install_utils import CondaPackage

ENVIRONMENT_FILE = Path(__file__).parents[2] / 'ClearMap3.yml'
QT_PACKAGE_NAME = 'qt6-main'
FIRST_QT_WITHOUT_THE_FONT_CRASH = Version('6.11.3')
MINIMUM_VERSION_OPERATOR = '>='


def test_the_environment_file_asks_for_a_qt_without_the_fontconfig_crash():
    dependencies = yaml.safe_load(ENVIRONMENT_FILE.read_text(encoding='utf-8'))['dependencies']
    qt_versions = [CondaPackage(dependency).version for dependency in dependencies
                   if isinstance(dependency, str) and CondaPackage(dependency).name == QT_PACKAGE_NAME]

    assert len(qt_versions) == 1, f'expected one {QT_PACKAGE_NAME} line in {ENVIRONMENT_FILE.name}, got {qt_versions}'
    version = qt_versions[0]
    assert version.startswith(MINIMUM_VERSION_OPERATOR), f'expected a minimum version, got {version!r}'
    assert Version(version.removeprefix(MINIMUM_VERSION_OPERATOR)) >= FIRST_QT_WITHOUT_THE_FONT_CRASH
