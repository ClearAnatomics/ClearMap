import argparse
from pathlib import Path

from ClearMap import Settings as settings
from ClearMap.config.atlas import DERIVED_ATLAS_NAME_RE, get_atlas_cache_dir, get_decompressed_atlases_dir
from ClearMap.config.early_boot import MachineConfig


def get_derived_atlas_folders():
    """The folders that can hold derived atlases: where the former versions wrote them, and where they are now."""
    return [Path(settings.atlas_folder),
            get_decompressed_atlases_dir(),  # when no target directory is given, next to the source atlas
            get_atlas_cache_dir(MachineConfig().get('atlas_cache_folder'))]


def find_derived_atlases():
    return [path for folder in get_derived_atlas_folders() if folder.is_dir()
            for path in sorted(folder.iterdir()) if DERIVED_ATLAS_NAME_RE.match(path.name)]


def clear_atlas_caches(verbose=False):
    atlas_cached_paths = find_derived_atlases()

    if verbose:
        from tqdm import tqdm
        print(f'Found {len(atlas_cached_paths)} files, deleting')

        for f_path in tqdm(atlas_cached_paths):
            f_path.unlink()
    else:
        for f_path in atlas_cached_paths:
            f_path.unlink()


def main():
    parser = argparse.ArgumentParser(prog='clearmap-utils',
                                     description='Set of CLI utilities for ClearMap',
                                     epilog='Example: clearmap-utils -v --clear-atlas-cache')

    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {0.1}')

    parser.add_argument('-x', '--clear-atlas-cache', dest='clear_atlas_cache', action='store_true',
                        help='Removes the atlases that are not default to save some space on disk.'
                             'These will be recomputed as required')
    parser.add_argument('-v', '--verbose', action='store_true', help='Turns on verbose mode.')

    args = parser.parse_args()

    if args.clear_atlas_cache:
        clear_atlas_caches(args.verbose)
    else:  # No options given
        parser.print_usage()


if __name__ == '__main__':
    main()
