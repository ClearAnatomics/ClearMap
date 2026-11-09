"""
Which commit of ClearMap is running, for the About dialog and bug reports.

An installed ClearMap (``python setup.py install`` copies it to site-packages) is not in a git repository: setup.py
records the commit of the checkout it installs in a stamp next to the installed code (``write_commit_stamp``), and
``get_commit_info`` reads it. A ClearMap run from its checkout has no stamp and asks git, so it follows ``git pull``.
A ClearMap that is neither (a zip, a copy) has no commit.
"""
import json
import logging
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_DIRECTORY = Path(__file__).resolve().parents[1]
GIT_TIMEOUT_SECONDS = 5
SHORT_HASH_LENGTH = 10
STAMP_FILE_NAME = '_commit_stamp.json'
DETACHED_HEAD_LABEL = 'detached HEAD'


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    commit_date: str  # YYYY-MM-DD
    branch: str | None  # None on a detached HEAD

    def __str__(self):
        return f'{self.commit_hash[:SHORT_HASH_LENGTH]} ({self.branch or DETACHED_HEAD_LABEL}, {self.commit_date})'


def _run_git(directory, *arguments):
    """The output of ``git <arguments>`` in ``directory``, or None if git cannot answer (no git, not a repository)."""
    try:
        completed = subprocess.run(['git', '-C', str(directory), *arguments], capture_output=True, text=True,
                                   check=True, timeout=GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as err:
        logger.debug('git %s failed in %s: %s', ' '.join(arguments), directory, err)
        return None
    return completed.stdout.strip()


def write_commit_stamp(commit_info, package_directory):
    """Record ``commit_info`` in ``package_directory``, which is then the installed ClearMap package."""
    (Path(package_directory) / STAMP_FILE_NAME).write_text(json.dumps(asdict(commit_info)), encoding='utf-8')


def _read_commit_stamp(package_directory):
    path = Path(package_directory) / STAMP_FILE_NAME
    if not path.exists():
        return None
    try:
        return CommitInfo(**json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, TypeError) as err:  # JSON errors are ValueErrors, wrong fields TypeErrors
        logger.warning('The commit stamp %s cannot be read (%s: %s), asking git instead', path, type(err).__name__, err)
        return None


def _read_commit_from_git(package_directory):
    """The commit of the repository whose root is the parent of ``package_directory``, if it is one."""
    repository_root = _run_git(package_directory, 'rev-parse', '--show-toplevel')
    if not repository_root or Path(repository_root).resolve() != Path(package_directory).resolve().parent:
        return None  # Not a checkout, or a package installed inside somebody else's repository
    commit = _run_git(package_directory, 'log', '-1', '--format=%H%n%cs')
    if not commit:
        return None
    commit_hash, commit_date = commit.splitlines()
    branch = _run_git(package_directory, 'branch', '--show-current')  # Empty on a detached HEAD
    return CommitInfo(commit_hash=commit_hash, commit_date=commit_date, branch=branch or None)


def get_commit_info(package_directory=PACKAGE_DIRECTORY) -> CommitInfo | None:
    """
    The commit of ClearMap: the one recorded when it was installed, or else the one that is checked out.

    Parameters
    ----------
    package_directory : str | Path
        The ClearMap package folder (the running one by default).

    Returns
    -------
    CommitInfo | None
        None if ClearMap was neither installed from a git checkout nor is one (or git is not installed).
    """
    return _read_commit_stamp(package_directory) or _read_commit_from_git(package_directory)
