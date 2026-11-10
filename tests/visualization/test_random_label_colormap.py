"""
The random colormap of label images (the DataViewer LUTs) has one colour per label.

It read the colours of the qualitative matplotlib colormap with matplotlib.cm.get_cmap, which matplotlib 3.9 removed:
the label LUTs failed with an AttributeError.
"""
import numpy as np

from ClearMap.Visualization.Color.Color import rand_cmap

BLACK = [0, 0, 0]


def test_the_random_label_colormap_has_one_colour_per_label_and_the_first_is_black():
    colors = rand_cmap(15, map_type='bright', first_color_black=True, cmap_name='Paired')

    assert colors.shape == (15, 3)
    assert colors[0].tolist() == BLACK
    assert not np.array_equal(colors[1], BLACK)


def test_the_pastel_label_colormap_repeats_its_colours_beyond_their_number():
    colors = rand_cmap(25, map_type='pastel', first_color_black=False, cmap_name='Set3')  # Set3 has 12 colours

    assert np.array_equal(colors[0], colors[12])
