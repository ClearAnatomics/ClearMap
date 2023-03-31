import os
import argparse
import glob

from tqdm import tqdm

from ClearMap.Settings import atlas_folder


def clear_atlas_caches(verbose=False):
    atlas_cached_paths = glob.glob(f'{atlas_folder}{os.sep}ABA_*None*.tif')

    if verbose:
        print(f'Found {len(atlas_cached_paths)} files, deleting')

        for f_path in tqdm(atlas_cached_paths):
            os.remove(f_path)
    else:
        for f_path in atlas_cached_paths:
            os.remove(f_path)


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
