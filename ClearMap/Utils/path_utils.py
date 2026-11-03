"""
Utility functions for working with paths
"""
import os
from pathlib import Path


def clean_path(path: str | Path) -> str:
    """Expand user (~) and normalize path."""
    return os.path.normpath(os.path.expanduser(str(path)))


def de_duplicate_path(root: Path, sub_dir) -> Path:
    """
    Remove sub_dir from root if root already ends with sub_dir

    Parameters
    ----------
    root: Path
        The root path
    sub_dir: Path
        The sub directory to check

    Returns
    -------
    Path
        The cleaned sub directory
    """
    if sub_dir != Path():
        sub_dir_parts = sub_dir.parts
        if root.parts[-len(sub_dir_parts):] == sub_dir_parts:
            sub_dir = Path()
    return sub_dir
