import argparse

from scipy import ndimage

from ClearMap.IO import IO as clearmap_io


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('annotation_file_path', type=str)
    parser.add_argument('distance_file_path', type=str)
    args = parser.parse_args()
    return args.annotation_file_path, args.distance_file_path


def main():
    annotation_file_path, dest_path = get_args()
    brain_mask = (clearmap_io.read(annotation_file_path) > 1).astype(int)
    distance_array = ndimage.distance_transform_edt(brain_mask)
    clearmap_io.write(dest_path, distance_array)


if __name__ == '__main__':
    main()
