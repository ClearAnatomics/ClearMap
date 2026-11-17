"""
``bin_props()`` histograms every property of the regions with the bins of the histogram of all the regions.

The bins were passed as ``groupby().agg({column: [np.histogram]}, bins=bins)``: the keywords of ``agg()`` are not
given to ``np.histogram`` (neither in pandas 1.5 nor in pandas 3: it was called with its default 10 bins), and the
bins were read back from the result of a first ``agg()`` that does not return them.
"""
import numpy as np
import pandas as pd

from ClearMap.Scripts.vasculature_stats_scripts import bin_props

N_BINS = 2
COLUMN = 'length'
REGIONS = {1: [1., 2.], 2: [3., 4.]}
ALL_VALUES_EDGES = [1., 2.5, 4.]  # 2 bins over the values of all the regions


def _graph_props():
    return {'edge_props': pd.DataFrame({'region_id': [1, 1, 2, 2], COLUMN: [1., 2., 3., 4.]})}


def test_the_bins_are_the_edges_of_the_histogram_of_all_the_regions():
    _, bins = bin_props(_graph_props(), 'edge_props', N_BINS, normed=False)

    np.testing.assert_allclose(bins[COLUMN].to_numpy(), ALL_VALUES_EDGES)


def test_each_region_is_histogrammed_with_these_bins():
    histograms, _ = bin_props(_graph_props(), 'edge_props', N_BINS, normed=False)

    for region, values in REGIONS.items():
        counts, edges = histograms.loc[region, (COLUMN, 'histogram')]
        expected_counts, _ = np.histogram(values, bins=ALL_VALUES_EDGES)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(edges, ALL_VALUES_EDGES)


def test_the_histograms_are_normalised_when_asked():
    histograms, _ = bin_props(_graph_props(), 'edge_props', N_BINS, normed=True)

    counts, edges = histograms.loc[1, (COLUMN, 'histogram')]
    assert np.sum(counts * np.diff(edges)) == 1.0
