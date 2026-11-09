"""
Which commit of ClearMap is running, for the About dialog and bug reports.

An installed ClearMap (``python setup.py install`` copies it to site-packages) is not in a git repository: setup.py
records the commit of the checkout it installs in a stamp next to the installed code (``write_commit_stamp``), and
``get_commit_info`` reads it. A ClearMap run from its checkout has no stamp and asks git, so it follows ``git pull``.
The zip of a commit made by ``git archive`` knows its commit too (``_git_archive.txt``, see ``.gitattributes``). A
ClearMap that is none of these (a copy, a modified zip) has no commit.
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
ARCHIVE_FILE_NAME = '_git_archive.txt'  # Its placeholders are replaced by git archive, see .gitattributes
ARCHIVE_PLACEHOLDER_MARKER = '$Format:'  # Left as is when the file is not in an archive
ARCHIVE_HEAD_PREFIX = 'HEAD -> '
ARCHIVE_TAG_PREFIX = 'tag: '
DETACHED_HEAD_LABEL = 'detached HEAD'


@dataclass(frozen=True)
class CommitInfo:
    commit_hash: str
    commit_date: str  # YYYY-MM-DD
    branch: str | None  # None on a detached HEAD
    tag: str | None = None  # The tag of the commit, e.g. of a release

    def __str__(self):
        short_hash = self.commit_hash[:SHORT_HASH_LENGTH]
        # Info priority: tag first, then branch, then commit ID
        if self.tag:
            ref_label = f'tag {self.tag}'
        elif self.branch:
            ref_label = self.branch
        else:
            ref_label = DETACHED_HEAD_LABEL
        return f'{short_hash} ({ref_label}, {self.commit_date})'

    def __str__(self):
        ref = self.branch or (f'tag {self.tag}' if self.tag else DETACHED_HEAD_LABEL)
        return f'{self.commit_hash[:SHORT_HASH_LENGTH]} ({ref}, {self.commit_date})'


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
    tag = _run_git(package_directory, 'describe', '--tags', '--exact-match')  # Fails if the commit has no tag
    return CommitInfo(commit_hash=commit_hash, commit_date=commit_date, branch=branch or None, tag=tag or None)


def _read_commit_of_archive(package_directory):
    """
    The commit that ``git archive`` wrote in ``ARCHIVE_FILE_NAME`` (hash, date and ref names, one per line).

    Git replaces the placeholders only when it builds the archive: it tells which commit the archive was made of, not
    whether the files were changed afterwards.
    """
    path = Path(package_directory) / ARCHIVE_FILE_NAME
    if not path.exists():
        return None
    lines = path.read_text(encoding='utf-8').splitlines()
    if len(lines) < 3 or any(ARCHIVE_PLACEHOLDER_MARKER in line for line in lines):
        return None  # Not in an archive: the placeholders are still there
    commit_hash, commit_date, ref_names = lines[0].strip(), lines[1].strip(), lines[2].split(',')
    refs = [ref.strip() for ref in ref_names]
    branch = next((ref[len(ARCHIVE_HEAD_PREFIX):] for ref in refs if ref.startswith(ARCHIVE_HEAD_PREFIX)), None)
    tag = next((ref[len(ARCHIVE_TAG_PREFIX):] for ref in refs if ref.startswith(ARCHIVE_TAG_PREFIX)), None)
    return CommitInfo(commit_hash=commit_hash, commit_date=commit_date, branch=branch, tag=tag)


def get_commit_info(package_directory=PACKAGE_DIRECTORY) -> CommitInfo | None:
    """
    The commit of ClearMap: the one recorded when it was installed, else the one that is checked out, else the one
    of the archive it was unpacked from.

    Parameters
    ----------
    package_directory : str | Path
        The ClearMap package folder (the running one by default).

    Returns
    -------
    CommitInfo | None
        None if ClearMap was installed from, is, or was unpacked from none of a git checkout or an archive of a commit
        (or git is not installed).
    """
    return (_read_commit_stamp(package_directory)
            or _read_commit_from_git(package_directory)
            or _read_commit_of_archive(package_directory))
