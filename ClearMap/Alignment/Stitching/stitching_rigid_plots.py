"""Plot functions and Layout plot mixin for StitchingRigid.

``LayoutPlotMixin`` is injected into ``Layout`` at runtime via ``@lazy_mixin``,
so matplotlib is never imported on a headless cluster.

Standalone functions duck-type on layout / source / alignment objects and only
import from ``StitchingRigid`` locally when they need to *construct* one.
"""

from __future__ import annotations

import warnings

import numpy as np

from ClearMap.Alignment.Stitching.layout_graph_utils import get_color_ids


class LayoutPlotMixin:
    """Plot methods for :class:`Layout`, injected via ``@lazy_mixin``."""

    def plot(self, **kwargs):
        """Overlay sources and display the result."""
        return plot_layout(self, **kwargs)

    def overlay(self, **kwargs):
        """Return a colour-overlay array."""
        return overlay_layout(self, **kwargs)

    def plot_regions(self, **kwargs):
        """Plot overlap regions."""
        return plot_regions(self.embedding()[2], sources=self.sources, **kwargs)

    def plot_alignments(self, **kwargs):
        """Plot alignment edges coloured by quality."""
        return plot_alignments(self.alignments, sources=self.sources, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Standalone plot helpers — duck-typed, no StitchingRigid dependency
# ═══════════════════════════════════════════════════════════════════════════

def plot_layout(layout, colors=None, percentile=98, normalize=True, color_ids=None):
    """Overlays and plots sources in a layout to check alignment.

    Arguments
    ---------
    layout : Layout class
        The layout to use for plotting.
    colors : list of colors or None
        The optional RGB colors to use.
    percentile : int
        Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
        If True normalize image to floats between 0 and 1.

    Returns
    -------
    image : array
        A color image of the overlayed sources.
    """
    img = overlay_layout(layout, colors=colors, percentile=percentile,
                         normalize=normalize, color_ids=color_ids)
    if img.ndim == 3:
        from matplotlib import pyplot as plt
        plt.imshow(np.transpose(img, [1, 0, 2])[:, :, :], origin='lower')
        plt.tight_layout()
    else:
        import ClearMap.Visualization.Plot3d as p3d
        p3d.plot(img)


def overlay_layout(layout, colors=None, percentile=98, normalize=True, color_ids=None):
    """Overlays the sources to check their placement.

    Arguments
    ---------
    layout : Layout class
        The layout with the sources to overlay.
    colors : list of tuple of floats or color names
        The optional RGB colors to use.
    percentile : int
        Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
        If True normalize image to floats between 0 and 1.

    Returns
    -------
    image : array
        A color image of the overlayed sources.
    """
    # full shape
    full_shape = tuple(layout.extent)
    full_lower = layout.lower

    source_colors = layout_coloring(layout, colors=colors, color_ids=color_ids)
    if colors == 'ids':
        image = [np.zeros(full_shape) for _ in range(max(source_colors) + 1)]
    else:
        image = np.zeros(full_shape + (3,))

    # construct full image
    sources = layout.sources
    for s, c in zip(sources, source_colors):
        l = s.lower
        u = s.upper
        r = tuple(slice(ll - fl, uu - fl) for ll, uu, fl in zip(l, u, full_lower))
        if colors == 'ids':
            image[c][r] += s[:]
        else:
            r += (slice(None),)
            if normalize:
                image[r] += np.multiply.outer(np.array(s[:], dtype=float) / s[:].max(), c)
            else:
                image[r] += np.multiply.outer(s[:], c)

    if percentile is not None:
        if colors == 'ids':
            for i in image:
                p = np.percentile(i, percentile)
                i[i > p] = p
        else:
            p = np.percentile(image, percentile)
            image[image > p] = p

    if normalize:
        if colors == 'ids':
            for i in image:
                i /= i.max()
        else:
            for c in range(3):
                image[..., c] /= image[..., c].max()

    return image


def layout_coloring(layout, colors=None, color_ids=None):
    """Assign colours to sources based on adjacency colouring."""
    from ClearMap.Visualization import Color as col

    sources = layout.sources
    nsources = len(sources)

    color_ids = get_color_ids(sources, nsources, color_ids)

    if colors == 'ids':
        return color_ids

    ncols = np.max(color_ids) + 1
    if colors is None:
        if ncols <= 2:
            colors = [[1, 0, 1], [0, 1, 0]]
        elif ncols <= 4:
            colors = [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0], [0, 0, 0.5]]
        else:
            colors = [[0.25,  0,     0    ],
                      [0,     0.25,  0    ],
                      [0,     0,     0.25 ],
                      [0.25,  0.25,  0    ],
                      [0,     0.25,  0.25 ],
                      [0.25,  0,     0.25 ],
                      [0.125, 0.25,  0    ],
                      [0.125, 0,     0.25 ]]

    colors = [col.color(c, alpha=False, as_int=True) for c in colors]
    colors = np.pad(colors[:ncols], ((0, max(0, ncols - len(colors))), (0, 0)), 'wrap')
    return colors[color_ids]


def plot_regions(regions, sources=None, cmap=None, annotate=True, axes=[0, 1]):
    """Overlays and plots regions to check the alignment.

    Arguments
    ---------
    regions : list of Region classes
        The regions to plot.
    sources : list of Source or None
        Sources used for annotation ids.
    cmap : colormap
        The color map to use to color the regions.
    annotate : bool
        Use annotation or not.
    """
    from matplotlib import pyplot as plt

    if cmap is None:
        cmap = plt.cm.rainbow

    if len(regions) == 0:
        return

    if sources is None:
        sources = list(np.unique(np.hstack([r.sources for r in regions])))

    sources_to_ids = {s: i for i, s in enumerate(sources)}

    ndim = regions[0].ndim
    if ndim != 2:
        warnings.warn(f"The regions are plotted in 2d using axes {axes!r} but are {ndim:d}d!")

    if axes is None:
        axes = [0, 1]

    ax = plt.gca()
    rmin = np.zeros(ndim)
    rmax = np.zeros(ndim)
    for i, r in enumerate(regions):
        rec = plt.Rectangle(np.array(r.lower)[axes],
                            r.upper[axes[0]] - r.lower[axes[0]],
                            r.upper[axes[1]] - r.lower[axes[1]],
                            fill=True, alpha=0.3,
                            color=cmap(float(i) / len(regions)))
        ax.add_patch(rec)
        if annotate:
            ids = [sources_to_ids[s] for s in r.sources]
            ax.annotate(str(tuple(ids)), xy=rec.get_xy(), xytext=(0, 0),
                        textcoords='offset points', color='w', ha='center', fontsize=8,
                        bbox=dict(boxstyle='round, pad=.5', fc=(.1, .1, .1, .92),
                                  ec=(1., 1., 1.), lw=1, zorder=1))
        rmin = np.min([rmin, r.lower], axis=0)
        rmax = np.max([rmax, r.upper], axis=0)

    plt.xlim((rmin[axes[0]], rmax[axes[0]]))
    plt.ylim((rmin[axes[1]], rmax[axes[1]]))


def plot_alignments(alignments, sources=None, axes=[0, 1], annotate=True,
                    min_quality=-np.inf, cmap=None):
    """Plots the alignments with their quality."""
    from matplotlib import pyplot as plt

    if cmap is None:
        cmap = plt.cm.hot

    ndim = alignments[0].ndim
    if ndim != 2:
        warnings.warn(f"The regions are plotted in 2d using axes {axes!r} but are {ndim:d}d!")
        ndim = 2

    if axes is None:
        axes = [0, 1]

    q = np.array([a.quality for a in alignments])

    q_max = np.max(q)
    if q_max == -np.inf:
        q_max = 0
        q_min = -1
    else:
        q_min = np.min(q)
        if q_min == -np.inf:
            q_min = np.min(q[q > -np.inf])
        q_min = max(min_quality, q_min)
        if q_max <= q_min:
            q_max = q_min + 1

    if sources is None:
        from ClearMap.Alignment.Stitching.StitchingRigid import sources_from_alignments
        sources = sources_from_alignments(alignments)

    # plot
    ax = plt.gca()
    rmin = np.zeros(ndim)
    rmax = np.zeros(ndim)

    for s in sources:
        # plot the source boundary
        lower = np.array(s.lower)[axes]
        upper = np.array(s.upper)[axes]

        rec = plt.Rectangle(lower, upper[0] - lower[0], upper[1] - lower[1],
                            fill=True, alpha=0.3, color='gray')
        ax.add_patch(rec)
        rmin = np.min([rmin, lower], axis=0)
        rmax = np.max([rmax, upper], axis=0)

    plt.xlim((rmin[axes[0]], rmax[axes[0]]))
    plt.ylim((rmin[axes[1]], rmax[axes[1]]))

    for a in alignments:
        p1 = 0.5 * (np.array(a.pre.lower)[axes] + np.array(a.pre.upper)[axes])
        p2 = 0.5 * (np.array(a.post.lower)[axes] + np.array(a.post.upper)[axes])
        if a.quality > -np.inf:
            c = cmap((float(a.quality) - q_min) / (q_max - q_min))
        else:
            c = 'black'

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c, linewidth=1)
        if annotate:
            ax.annotate(f'{a.quality:.2e}', xy=0.5 * (p1 + p2), xytext=(0, 0),
                        textcoords='offset points', color='w', ha='center', fontsize=8,
                        bbox=dict(boxstyle='round, pad=.5', fc=(.1, .1, .1, .92),
                                  ec=(1., 1., 1.), lw=1, zorder=1))


# ═══════════════════════════════════════════════════════════════════════════
# Convenience wrappers — construct SR objects, so local imports required
# ═══════════════════════════════════════════════════════════════════════════

def plot_sources(sources, colors=None, percentile=98, normalize=True):
    """Overlays and plots sources in a layout to check alignment.

    Arguments
    ---------
    sources : list of Source
        The sources to overlay.
    colors : list of colors or None
        The optional RGB colors to use.
    percentile : int
        Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
        If True normalize image to floats between 0 and 1.

    Returns
    -------
    image : array
        A color image of the overlayed sources.
    """
    from ClearMap.Alignment.Stitching.StitchingRigid import Layout
    layout = Layout(sources=sources)
    return plot_layout(layout, colors=colors, percentile=percentile, normalize=normalize)


def overlay_sources(sources, colors=None, percentile=98, normalize=True):
    """Overlays the sources to check their placement.

    Arguments
    ---------
    sources : list of Source
        The sources to overlay.
    colors : list of tuple of floats or color names
        The optional RGB colors to use.
    percentile : int
        Use this percentile as upper cutoff in the resulting image to enhance contrast.
    normalize : bool
        If True normalize image to floats between 0 and 1.

    Returns
    -------
    image : array
        A color image.
    """
    from ClearMap.Alignment.Stitching.StitchingRigid import Layout
    layout = Layout(sources=sources)
    return overlay_layout(layout, colors=colors, percentile=percentile, normalize=normalize)


def layout_along_axis_mip(src1, src2, axis=2, depth=10, max_shifts=10, ranges=None, verbose=False):
    """Layout corresponding to a MIP-projected alignment."""
    from ClearMap.Alignment.Stitching.StitchingRigid import (Layout, Source, Slice, _format_max_shifts, _mip_axis,
                                                             max_intensity_projection)

    # format the shifts
    ndim = src1.ndim
    if not isinstance(depth, (list, tuple)):
        depth = (depth,) * ndim

    if not isinstance(ranges, list):
        ranges = [ranges] * ndim

    max_shifts = _format_max_shifts(max_shifts, ndim)
    mip_axis = _mip_axis(src1, src2, axis=None, max_shifts=max_shifts)
    mip_depth = depth[mip_axis]

    # reduce sources to ranges along non-mip axes
    if ranges != [None] * len(ranges):
        sl1 = ()
        sl2 = ()
        p1 = src1.position
        p2 = src2.position
        for d, r in enumerate(ranges):
            if d != mip_axis and r is not None:
                sl1 += (slice(r[0] - p1[d], r[1] - p1[d]),)
                sl2 += (slice(r[0] - p2[d], r[1] - p2[d]),)
            else:
                sl1 += (slice(None),)
                sl2 += (slice(None),)

        src1 = Slice(source=src1, slicing=sl1)
        src2 = Slice(source=src2, slicing=sl2)

    # mip
    mip_depth = depth[mip_axis]
    max_shifts = max_shifts[:mip_axis] + max_shifts[mip_axis + 1:]

    s1 = src1.shape
    s2 = src2.shape

    # max intensity projections
    sub1 = [slice(None)] * ndim
    sub1[mip_axis] = slice(max(0, s1[mip_axis] - mip_depth), None)
    sub1 = tuple(sub1)

    sub2 = [slice(None)] * ndim
    sub2[mip_axis] = slice(None, min(mip_depth, s2[mip_axis]))
    sub2 = tuple(sub2)

    # calculate max projection along axis
    mip1 = max_intensity_projection(src1[sub1], axis=mip_axis)
    mip2 = max_intensity_projection(src2[sub2], axis=mip_axis)

    # add position information
    p1 = src1.position[:mip_axis] + src1.position[mip_axis + 1:]
    p2 = src2.position[:mip_axis] + src2.position[mip_axis + 1:]

    mip1 = Source(mip1, position=p1, tile_position=src1.tile_position)
    mip2 = Source(mip2, position=p2, tile_position=src2.tile_position)

    return Layout(sources=[mip1, mip2])


def plot_along_axis_mip(src1, src2, axis=2, depth=10, max_shifts=10, ranges=None, verbose=False, **kwargs):
    layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth,
                                   max_shifts=max_shifts, ranges=ranges, verbose=verbose)
    return plot_layout(layout, **kwargs)


def overlay_along_axis_mip(src1, src2, axis=2, depth=10, max_shifts=10, ranges=None, verbose=False, **kwargs):
    layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth, max_shifts=max_shifts, ranges=ranges, verbose=verbose)
    return overlay_layout(layout, **kwargs)
