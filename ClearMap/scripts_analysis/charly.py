import os
import sys

import numpy as np

from tifffile import tifffile


src_folder_path = sys.argv[1]
img_base_name = sys.argv[2]
dest_folder_path = sys.argv[3]
channel_base_name = "_C0{}.ome.tif"
if len(sys.argv) > 4:
    channel_id = sys.argv[4]
else:
    channel_id = 0


def extract_corner(img, i, j, new_width=128, new_height=128, start_z=2000, end_z=2500):
    start_x, end_x, start_y, end_y = get_corner_coords(img, i, j, new_height, new_width)
    return img[start_x:end_x, start_y:end_y, start_z, end_z]


def get_corner_coords(img, i, j, new_width=128, new_height=128):
    if img.ndim == 2:
        width, height = img.shape
    elif img.ndim == 3:
        width, height, depth = img.shape
    else:
        raise ValueError("Expected a 2D or 3D image")
    start_x = (width - new_width) if i == 0 else 0
    start_y = (height - new_height) if j == 0 else 0
    end_x = -1 if i == 0 else new_width
    end_y = -1 if j == 0 else new_height
    return start_x, end_x, start_y, end_y


def extract_images():
    for i in range(2):
        for j in range(2):
            image_name = "{}{}{}".format(img_base_name,
                                         "[{} x {}]".format(str(i).zfill(2), str(j).zfill(2)),
                                         channel_base_name.format(channel_id))
            image_path = os.path.join(src_folder_path, image_name)
            img = tifffile.imread(image_path)
            new_image_name = "test_{}{}".format(
                "[{} x {}]".format(str(i).zfill(2), str(j).zfill(2)),
                channel_base_name.format(channel_id)
            )
            tifffile.imsave(os.path.join(dest_folder_path, new_image_name), extract_corner(img, i, j))


def test():
    for i in range(2):
        for j in range(2):
            image_name = "{}{}{}".format(img_base_name,
                                         "[{} x {}]".format(str(i).zfill(2), str(j).zfill(2)),
                                         channel_base_name.format(channel_id))
            image_path = os.path.join(src_folder_path, image_name)
            print(image_path)
            print(get_corner_coords(np.empty((1024, 1024, 256)), i, j))


if __name__ == '__main__':
    # test()
    extract_images()