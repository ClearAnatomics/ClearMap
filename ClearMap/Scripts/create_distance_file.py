import argparse

from scipy import ndimage

from ClearMap.Alignment.Annotation import annotation_to_distance_file
from ClearMap.IO import IO as clearmap_io


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotation_file_path', type=str)
    parser.add_argument('distance_file_path', type=str)
    args = parser.parse_args()
    return args.annotation_file_path, args.distance_file_path


def main():
    annotation_file_path, dest_path = get_args()
    clearmap_io.write(dest_path, annotation_to_distance_file(annotation_file_path))


if __name__ == '__main__':
    main()
