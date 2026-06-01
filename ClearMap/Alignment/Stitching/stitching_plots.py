"""
stitching_plots
===============

Plot functions and Layout/Alignment plot mixins for rigid and wobbly stitching.

Mixins are injected at runtime via ``@lazy_mixin``, so matplotlib is never
imported on a headless cluster unless a plot method is actually called.

Standalone functions duck-type on layout / source / alignment objects and only
import from ``stitching_rigid`` / ``stitching_wobbly`` locally when they need to
*construct* helper objects.
"""

from __future__ import annotations

import warnings

import numpy as np

from ClearMap.Alignment.Stitching.layout_graph_utils import get_color_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Rigid Layout mixin
# ═══════════════════════════════════════════════════════════════════════════════

class LayoutPlotMixin:
    """Plot methods for :class:`Layout`, injected via ``@lazy_mixin``."""

    def plot(self, colors=None, percentile=98, normalize=True, color_ids=None, coordinate=None, axis=2):
        """Plots overlayed sources to check their placement.

        Arguments
        ---------
        colors : list of tuple of floats or color names
          The optional RGB colors to use.
        percentile : int
          Use this percentile as upper cutoff in the resulting image to enhance contrast.
        normalize : bool
          If True normalize image to floats between 0 and 1.
        color_ids : list of ints
          Use specific color ids for the sources contributing to the layout.
        coordinate : int or None
          Optional coordinate at which to take a slice.
        axis : int
          Optional axis to take the slice in.

        Returns
        -------
        image : array
          A color image.
        """
        if coordinate is None:
            layout = self
        else:
            layout = self.slice_along_axis(coordinate=coordinate, axis=axis)
        return plot_layout(layout, colors=colors, percentile=percentile, normalize=normalize, color_ids=color_ids)

    def overlay(self, colors=None, percentile=98, normalize=True, coordinate=None, axis=2):
        if coordinate is None:
            layout = self
        else:
            layout = self.slice_along_axis(coordinate=coordinate, axis=axis)
        return overlay_layout(layout, colors=colors, percentile=percentile, normalize=normalize)

    def plot_regions(self, cmap=None, annotate=True, axes=None):
        """Overlays and plots regions to check the alignment of this layout.

        Arguments
        ---------
        cmap : colormap
          The color map to use to color the regions.
        annotate : bool
          Use annotation or not.
        axes : tuple of ints
          Axes to use if sources are larger than 2d.
        """
        if axes is None:
            axes = [0, 1]
        _, _, regions = self.embedding()
        return plot_regions(regions, sources=self.sources, cmap=cmap, annotate=annotate, axes=axes)

    def plot_alignments(self, cmap=None, annotate=True, axes=None):
        """Overlays and plots regions to check the alignment of this layout.

        Arguments
        ---------
        cmap : colormap
          The color map to use to color the regions.
        annotate : bool
          Use annotation or not.
        axes : tuple of ints
          Axes to use if sources are larger than 2d.
        """
        if axes is None:
            axes = [0, 1]
        if cmap is None:
            from matplotlib import pyplot as plt
            plt.cm.rainbow
        return plot_alignments(self.alignments, sources=self.sources, cmap=cmap, annotate=annotate, axes=axes)


# ═══════════════════════════════════════════════════════════════════════════════
# Rigid Alignment mixin
# ═══════════════════════════════════════════════════════════════════════════════

class AlignmentBasePlotMixin:
    """Plot methods for :class:`AlignmentBase`, injected via ``@lazy_mixin``."""

    def plot(self, *args, **kwargs):
        """Plot the two sources overlayed."""
        plot_sources(self.sources, *args, **kwargs)

    def overlay(self, **kwargs):
        """Return a colour-overlay array of the two sources."""
        return overlay_sources(self.sources, **kwargs)

    def plot_overlay(self, **kwargs):
        """Overlay and display with 3D viewer."""
        import ClearMap.Visualization.Plot3d as p3d
        ovl = self.overlay(colors='ids', **kwargs)
        return p3d.plot([ovl])

    def overlay_overlap(self, max_shifts=0):
        """Return the overlapping sub-arrays of pre and post."""
        from ClearMap.Alignment.Stitching.stitching_rigid import _overlap_with_shifts
        o1, o2 = _overlap_with_shifts(self.pre, self.post, max_shifts=max_shifts)
        i1 = self.pre[o1.local_slicing(self.pre)]
        i2 = self.post[o2.local_slicing(self.post)]
        return [i1, i2]

    def plot_overlap(self, **kwargs):
        """Display the overlap region with the 3D viewer."""
        import ClearMap.Visualization.Plot3d as p3d
        return p3d.plot([self.overlay_overlap(**kwargs)])

    def overlay_mip(self, *args, **kwargs):
        """Overlay using max intensity projection."""
        return overlay_along_axis_mip(self.pre, self.post, *args, **kwargs)

    def plot_mip(self, *args, **kwargs):
        """Plot using max intensity projection."""
        return plot_along_axis_mip(self.pre, self.post, *args, **kwargs)


class AlignmentPlotMixin(AlignmentBasePlotMixin):
    """Plot methods for :class:`Alignment`.

    Overrides ``plot`` to shift post by the computed displacement before
    overlaying.
    """

    def plot(self, *args, **kwargs):
        post = self.post.copy()
        post.position = tuple(p + d for p, d in zip(self.pre.position, self.displacement))
        return plot_sources([self.pre, post], *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Wobbly Alignment mixin
# ═══════════════════════════════════════════════════════════════════════════════

class WobblyAlignmentPlotMixin:
    """Plot methods for :class:`WobblyAlignment`, injected via ``@lazy_mixin``."""

    def plot_overlay_wobbly(self):
        """Display wobbly overlay with 3D viewer."""
        import ClearMap.Visualization.Plot3d as p3d
        return p3d.plot([self.overlay_wobbly()])

    def overlay_mip_wobbly(self, overlap=True, mip_axis=None,
                           percentile=98, normalize=True):
        """Colour MIP overlay of the wobbly alignment."""
        from ClearMap.Alignment.Stitching.stitching_rigid import _mip_axis
        import ClearMap.Visualization.Color as col

        ovl_mip = self.overlay_wobbly(overlap=overlap)

        # max project
        if mip_axis is None:
            mip_axis = _mip_axis(self.pre, self.post)

        ovl_mip = [np.max(o, axis=mip_axis) for o in ovl_mip]
        for o in ovl_mip:
            p = np.percentile(o, percentile)
            o[o > p] = p

        colors = [[1, 0, 1], [0, 1, 0]]
        colors = [col.color(c, alpha=False, as_int=True) for c in colors]

        image = np.zeros(ovl_mip[0].shape + (3,))
        for o, c in zip(ovl_mip, colors):
            image += np.multiply.outer(o / o.max(), c)

        if normalize:
            for c in range(3):
                image[:, :, c] /= image[:, :, c].max()

        return image

    def plot_mip_wobbly(self, overlap=True, mip_axis=None, percentile=98):
        """Display wobbly MIP overlay with matplotlib."""
        from matplotlib import pyplot as plt
        image = self.overlay_mip_wobbly(overlap=overlap, mip_axis=mip_axis, percentile=percentile)
        plt.imshow(np.transpose(image, [1, 0, 2])[:, :, :], origin='lower')
        plt.tight_layout()


# ═══════════════════════════════════════════════════════════════════════════════
# Wobbly Layout mixin
# ═══════════════════════════════════════════════════════════════════════════════

class WobblyLayoutPlotMixin:
    """Plot methods for :class:`WobblyLayout`, injected via ``@lazy_mixin``."""

    def plot_wobble(self):
        """Plot per-alignment displacement curves."""
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        axs = [plt.subplot(1, 2, i + 1) for i in range(2)]
        for i, a in enumerate(self.alignments):
            invalid = a.status < a.VALID
            d = np.array(a.displacements, dtype=float)
            d[invalid] = np.nan
            x = np.arange(a.lower_coordinate, a.upper_coordinate)
            label = f'{i:d}: {a.pre.identifier!r}-{a.post.identifier!r}'
            plt.subplot(1, 2, 1)
            plt.plot(x, d[:, 0], label=label)
            plt.subplot(1, 2, 2, sharex=axs[0])
            plt.plot(x, d[:, 1], label=label)

        def on_pick(event):
            for ax in axs:
                for curve in ax.get_lines():
                    if curve.contains(event)[0]:
                        print(curve.get_label())

        fig.canvas.mpl_connect('motion_notify_event', on_pick)
        plt.show()

    def alignment_info(self, tile_position, coordinate,
                       plot=True, use_displacements=True, **kwargs):
        """Diagnostic: gather alignment info and optionally overlay a slice."""
        import ClearMap.IO.Slice as slc
        from ClearMap.Alignment.Stitching.stitching_wobbly import WobblySource, WobblyAlignment
        from ClearMap.Alignment.Stitching.stitching_rigid import Source, Layout

        s = self.source_from_tile_position(tile_position)
        status = s.status_at_coordinate(coordinate)

        a_status = []
        alignments = self.alignments_from_tile_position(tile_position)
        for a in alignments:
            a_status.append((a.pre.identifier, a.post.identifier,
                             a.status_at_coordinate(coordinate)))

        print(f'Source status: {WobblySource.status_to_description[status]!r}')
        for a in a_status:
            status = WobblyAlignment.status_to_description[a[2]]
            print(f'Alignment status: {a[0]!r}->{a[1]!r}: {status !r}')

        # plot overlay to all neighbours
        sliced_layout = None
        if plot:
            axis = self.axis
            ndim = self.ndim
            sources = [s]
            for a in alignments:
                if a.pre not in sources:
                    sources.append(a.pre)
                if a.post not in sources:
                    sources.append(a.post)

            sliced_sources = []
            for source in sources:
                if use_displacements:
                    if source == s:
                        position = tuple(0 for _ in range(ndim - 1))
                    else:
                        for a in alignments:
                            if source == a.pre:
                                position = tuple(-p for p in a.displacement)
                                break
                            if source == a.post:
                                position = a.displacement
                                break
                        position = position[:axis] + position[axis + 1:]
                else:
                    position = source.wobble_at_coordinate(coordinate)

                slicing = ((slice(None),) * axis
                           + (coordinate - source.coordinate,)
                           + (slice(None),) * (ndim - 1 - axis))
                sliced_sources.append(
                    Source(source=slc.Slice(source=source.source.as_virtual(), slicing=slicing),
                           position=position, tile_position=source.tile_position))

            sliced_layout = Layout(sources=sliced_sources, shape=None, position=None,
                                   dtype=self.dtype, order=self.order)
            plot_layout(sliced_layout, **kwargs)

        return sliced_layout


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone plot helpers — duck-typed, no stitching_rigid import at module level
# ═══════════════════════════════════════════════════════════════════════════════

def plot_layout(layout, colors=None, percentile=98, normalize=True, color_ids=None):
    """Overlays and plots sources in a layout to check alignment.

    ..Note::
        For 2-D layouts uses matplotlib; for 3-D uses Plot3d.

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
    """Build a colour-overlay array of all sources in a layout.

    Arguments
    ---------
    layout : Layout
        The layout with the sources to overlay.
    colors : list of colours, ``'ids'``, or None
        RGB colours, or ``'ids'`` to return per-channel arrays.
    percentile : int
        Upper cutoff percentile for contrast enhancement.
    normalize : bool
        Normalize to [0, 1] floats.
    color_ids : list of int or None
        Explicit colour index per source.

    Returns
    -------
    image : array
        A colour image (or list of single-channel images when ``colors='ids'``).
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
                mx = i.max()
                if mx > 0:
                    i /= mx
        else:
            for c in range(3):
                mx = image[..., c].max()
                if mx > 0:
                    image[..., c] /= mx

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


def plot_regions(regions, sources=None, cmap=None, annotate=True, axes=None):
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

    if axes is None:
        axes = [0, 1]
    if cmap is None:
        cmap = plt.cm.rainbow

    if len(regions) == 0:
        return

    if sources is None:
        sources = list(np.unique(np.hstack([r.sources for r in regions])))

    sources_to_ids = {s: i for i, s in enumerate(sources)}

    ndim = regions[0].ndim
    if ndim != 2:
        warnings.warn(f"Regions are plotted in 2d using axes {axes!r} but are {ndim:d}d!")

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


def plot_alignments(alignments, sources=None, axes=None, annotate=True,
                    min_quality=-np.inf, cmap=None):
    """Plot alignment edges coloured by quality."""
    from matplotlib import pyplot as plt

    if axes is None:
        axes = [0, 1]
    if cmap is None:
        cmap = plt.cm.hot

    ndim = alignments[0].ndim
    if ndim != 2:
        warnings.warn(f"Regions are plotted in 2d using axes {axes!r} but are {ndim:d}d!")
        ndim = 2

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
        from ClearMap.Alignment.Stitching.stitching_rigid import sources_from_alignments
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


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrappers — construct SR objects, so local imports needed
# ═══════════════════════════════════════════════════════════════════════════════

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
    from ClearMap.Alignment.Stitching.stitching_rigid import Layout
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
    from ClearMap.Alignment.Stitching.stitching_rigid import Layout
    layout = Layout(sources=sources)
    return overlay_layout(layout, colors=colors, percentile=percentile, normalize=normalize)


def layout_along_axis_mip(src1, src2, axis=2, depth=10, max_shifts=10, ranges=None, verbose=False):
    """Build a Layout from MIP-projected sources."""
    from ClearMap.Alignment.Stitching.stitching_rigid import (Layout, Source, Slice, _format_max_shifts, _mip_axis,
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
    """Plot a MIP-projected alignment."""
    layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth,
                                   max_shifts=max_shifts, ranges=ranges, verbose=verbose)
    return plot_layout(layout, **kwargs)


def overlay_along_axis_mip(src1, src2, axis=2, depth=10, max_shifts=10, ranges=None, verbose=False, **kwargs):
    """Build a colour overlay of a MIP-projected alignment."""
    layout = layout_along_axis_mip(src1, src2, axis=axis, depth=depth, max_shifts=max_shifts, ranges=ranges, verbose=verbose)
    return overlay_layout(layout, **kwargs)
