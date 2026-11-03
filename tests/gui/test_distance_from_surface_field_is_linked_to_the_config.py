"""
The "Distance from surface" field of the cell filters page was linked to nothing: what was typed there never reached the
config, so the cell filter never used it. It is the setting cell_filtration.distance_from_surface_px of the channel.
"""
from test_channel_page_writes_no_config import CHANNEL_NAME, _collection, patches  # noqa: F401  (fixture)

from ClearMap.gui.params import CellMapParams


def test_the_distance_from_surface_field_is_the_cell_filtration_setting_of_the_channel(patches):  # noqa: F811
    collection = _collection(CellMapParams, patches, sample_params=None)
    collection.add_channel(CHANNEL_NAME)

    link = collection.channel_params[CHANNEL_NAME].params_dict['distance_from_surface_px']

    assert link.keys == ['cell_filtration', 'distance_from_surface_px']
    assert link.default == 0  # The config of a sample made before this setting existed has none
