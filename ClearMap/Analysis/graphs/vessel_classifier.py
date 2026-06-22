"""
vessel_classifier
=================

Iterative artery/vein classification on a reduced vascular graph.

The algorithm alternates between artery and vein refinement because each
vessel type is defined partly by *not being the other*:

    1. **Pre-filter arteries** — remove noise (tiny connected components)
       from the initial ``artery_binary`` mask.
    2. **Restrictive vein mask** — identify edges that are *definitely* veins:
       large radius + low artery marker + (optionally) high vein marker.
       These are excluded from the artery mask.
    3. **Permissive vein mask** — temporarily widen the vein definition to
       include any large non-artery edge.  Used as a boundary for artery tracing.
    4. **Trace arteries** — hysteresis extension: walk along edges that are
       large enough and have sufficient artery marker intensity, stopping at
       veins, brain surface, or low signal.
    5. **Final vein mask** — recompute after artery tracing stabilised.
    6. **Trace veins** — hysteresis extension, stopping before arteries.
    7. **Cleanup** — remove small artery/vein components (fragments that
       tracing extended but not enough to be biologically meaningful).

All thresholds are collected in :class:`ClassificationConfig` so the
classifier is fully parameterised and testable without touching the GUI.

See Also
--------
:class:`~ClearMap.pipeline_orchestrators.tube_map.VesselGraphProcessor`
    Orchestrator that builds the graph and calls this classifier.
[Kirst2020]_.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ClearMap.Analysis.graphs import graph_processing
from ClearMap.Analysis.graphs.graph_gt import Graph


@dataclass(frozen=True)
class ClassificationConfig:
    """
    All thresholds for the iterative artery/vein classification.

    Radius thresholds are unit-agnostic: they are in the same units as
    the radii in :class:`VesselSignals` (µm for current graphs, voxels
    for legacy graphs).  The ``from_config`` factory resolves the correct
    source depending on the ``use_legacy`` flag.

    Attributes
    ----------
    vein_intensity_range : tuple[float, float]
        (low, high) intensity range on the artery channel that
        characterises veins.  Edges *within* this range AND with
        large radius are classified as restrictive veins.
    restrictive_vein_radius : float
        Minimum radius for an edge to qualify as a
        *restrictive* (high-confidence) vein.
    permissive_vein_radius : float
        Minimum radius for the *permissive* vein mask,
        used as a temporary boundary during artery tracing.
    final_vein_radius : float
        Minimum radius for the *final* vein mask,
        computed after artery tracing has stabilised.
    arteries_min_noise_edges : int
        Minimum number of connected artery edges to survive
        initial noise removal (step 1).
    artery_intensity_min : float
        Minimum artery marker intensity for tracing to continue.
    artery_trace_radius : float
        Minimum edge radius to continue artery tracing.
    vein_trace_radius : float
        Minimum edge radius to continue vein tracing.
    vein_intensity_min : float
        Minimum vein marker intensity to continue vein tracing.
        Only used when a dedicated vein channel exists.
    distance_to_surface_min : float
        Minimum distance to brain surface (atlas voxels) for
        tracing to continue.
    max_artery_trace_iter : int
        Maximum hysteresis iterations for artery tracing.
    max_vein_trace_iter : int
        Maximum hysteresis iterations for vein tracing.
    min_artery_component_edges : int
        Minimum edges in a connected artery component to survive
        final cleanup (step 7).
    min_vein_component_edges : int
        Minimum edges in a connected vein component to survive
        final cleanup (step 7).
    """
    vein_intensity_range:        tuple[float, float]
    restrictive_vein_radius:     float
    permissive_vein_radius:      float
    final_vein_radius:           float
    arteries_min_noise_edges:    int
    artery_intensity_min:        float
    artery_trace_radius:         float
    vein_trace_radius:           float
    vein_intensity_min:          float
    distance_to_surface_min:     float
    max_artery_trace_iter:       int
    max_vein_trace_iter:         int
    min_artery_component_edges:  int
    min_vein_component_edges:    int

    @classmethod
    def from_config(cls, cfg: dict, legacy_thresholds: dict = None,
                    use_legacy: bool = False) -> 'ClassificationConfig':
        """
        Build from the ``vessel_type_postprocessing`` config subtree.

        Parameters
        ----------
        cfg : dict
            The ``vessel_type_postprocessing`` config subtree.
        legacy_thresholds : dict or None
            ``VesselGraphProcessor._LEGACY_THRESHOLDS`` dict.
            Required when ``use_legacy=True``.
        use_legacy : bool
            If True, read voxel-space thresholds from ``legacy_thresholds``
            instead of µm values from config.
        """
        pre = cfg['pre_filtering']
        tr  = cfg['tracing']
        cap = cfg['capillaries_removal']

        if use_legacy:
            if legacy_thresholds is None:
                raise ValueError('legacy_thresholds required when use_legacy=True')
            t = legacy_thresholds
            return cls(
                vein_intensity_range=tuple(pre['vein_intensity_range_on_arteries_ch']),
                restrictive_vein_radius=t['restrictive_vein_radius_vx'],
                permissive_vein_radius=t['permissive_vein_radius_vx'],
                final_vein_radius=t['final_vein_radius_vx'],
                arteries_min_noise_edges=pre['arteries_min_noise_edges'],
                artery_intensity_min=tr['artery_intensity_min'],
                artery_trace_radius=t['artery_trace_radius_vx'],
                vein_trace_radius=t['vein_trace_radius_vx'],
                vein_intensity_min=tr.get('vein_intensity_min', 200.0),
                distance_to_surface_min=tr['distance_to_surface_min'],
                max_artery_trace_iter=tr['max_arteries_iterations'],
                max_vein_trace_iter=tr['max_veins_iterations'],
                min_artery_component_edges=cap['min_artery_component_edges'],
                min_vein_component_edges=cap['min_vein_component_edges'],
            )

        return cls(
            vein_intensity_range=tuple(pre['vein_intensity_range_on_arteries_ch']),
            restrictive_vein_radius=pre['restrictive_vein_radius_um'],
            permissive_vein_radius=pre['permissive_vein_radius_um'],
            final_vein_radius=pre['final_vein_radius_um'],
            arteries_min_noise_edges=pre['arteries_min_noise_edges'],
            artery_intensity_min=tr['artery_intensity_min'],
            artery_trace_radius=tr['artery_trace_radius_um'],
            vein_trace_radius=tr['vein_trace_radius_um'],
            vein_intensity_min=tr.get('vein_intensity_min', 200.0),
            distance_to_surface_min=tr['distance_to_surface_min'],
            max_artery_trace_iter=tr['max_arteries_iterations'],
            max_vein_trace_iter=tr['max_veins_iterations'],
            min_artery_component_edges=cap['min_artery_component_edges'],
            min_vein_component_edges=cap['min_vein_component_edges'],
        )


@dataclass
class VesselSignals:
    """
    All per-edge arrays needed by the classifier, collected once before
    classification starts.

    Fields are ``None`` when the corresponding channel does not exist
    (e.g. no dedicated vein channel → ``vein_binary = None``).
    """
    radii:                np.ndarray                    # (n_edges,) — radii in whichever unit was supplied (should match thresholds)
    artery_binary:        Optional[np.ndarray] = None   # (n_edges,) — bool, from binary mask
    artery_intensity:     Optional[np.ndarray] = None   # (n_edges,) — float, raw marker signal
    vein_binary:          Optional[np.ndarray] = None   # (n_edges,) — bool, from binary mask
    vein_intensity:       Optional[np.ndarray] = None   # (n_edges,) — float, raw marker signal
    distance_to_surface:  Optional[np.ndarray] = None   # (n_edges,) — atlas voxels


class VesselClassifier:
    """
    Iterative artery/vein edge classifier for reduced vascular graphs.

    Usage
    -----
    ::

        signals = VesselSignals(...)
        cfg     = ClassificationConfig.from_config(config_dict)
        vc      = VesselClassifier(graph, cfg, signals)
        vc.classify()
        # graph now has 'artery' and 'vein' edge properties

    Parameters
    ----------
    graph : Graph
        The annotated, reduced graph.  Modified **in place** —
        ``'artery'`` and ``'vein'`` edge properties are defined/updated.
    config : ClassificationConfig
        All classification thresholds.
    signals : VesselSignals
        Pre-collected per-edge signal arrays.
    """

    def __init__(self, graph: Graph, config: ClassificationConfig,
                 signals: VesselSignals):
        self.graph   = graph
        self.cfg     = config
        self.signals = signals

    # ── public entry point ───────────────────────────────────────────────

    def classify(self) -> None:
        """
        Run the full iterative classification pipeline.

        After this call the graph has two boolean edge properties:

        * ``'artery'`` — True for edges classified as artery
        * ``'vein'``   — True for edges classified as vein

        Edges that are neither artery nor vein are implicitly capillaries.
        """
        cfg = self.cfg

        # Step 1 — noise removal on artery_binary
        artery = self._pre_filter_arteries()

        # Step 2 — restrictive (high-confidence) vein mask
        restrictive_veins = self._restrictive_vein_mask()
        artery[restrictive_veins] = False

        self.graph.define_edge_property('artery', artery)

        # Step 3 — permissive vein mask (temporary, for artery tracing boundary)
        tmp_veins = self._vein_mask(restrictive_veins, artery,
                                    cfg.permissive_vein_radius)

        # Step 4 — trace arteries
        self._trace_arteries(tmp_veins)

        # Step 5 — final vein mask (after artery tracing stabilised)
        vein = self._vein_mask(restrictive_veins,
                               self.graph.edge_property('artery'),
                               cfg.final_vein_radius)
        self.graph.define_edge_property('vein', vein)

        # Step 6 — trace veins
        self._trace_veins()

        # Step 7 — cleanup small components
        self._remove_small_components('artery', cfg.min_artery_component_edges)
        self._remove_small_components('vein',   cfg.min_vein_component_edges)

    def _pre_filter_arteries(self) -> np.ndarray:
        """
        Remove tiny connected artery components (noise from the binary mask).

        Returns a copy of ``artery_binary`` with components smaller than
        ``arteries_min_noise_edges`` set to False.
        """
        artery = self.signals.artery_binary.copy()

        sub = self.graph.sub_graph(edge_filter=artery, view=True)
        sub_edge, edge_map = sub.edge_graph(return_edge_map=True)
        components, sizes = sub_edge.label_components(return_vertex_counts=True)

        too_small = edge_map[
            np.isin(components,
                    np.where(sizes < self.cfg.arteries_min_noise_edges)[0])]
        artery[too_small] = False
        return artery

    def _restrictive_vein_mask(self) -> np.ndarray:
        """
        High-confidence vein identification: edges that are large AND have
        vein-compatible signal on both channels (when available).

        Uses both positive evidence (vein channel) and negative evidence
        (low artery channel) when both are available.
        """
        s   = self.signals
        cfg = self.cfg

        large = s.radii >= cfg.restrictive_vein_radius

        if s.artery_intensity is not None:
            lo, hi = cfg.vein_intensity_range
            in_vein_range = (s.artery_intensity >= lo) & (s.artery_intensity <= hi)
            if s.vein_intensity is not None:
                high_vein  = s.vein_intensity >= cfg.vein_intensity_min
                return in_vein_range & high_vein & large
            else:
                return in_vein_range & large
        else:
            return large.copy()

    def _vein_mask(self, restrictive_veins: np.ndarray,
                   artery: np.ndarray, radius_threshold: float) -> np.ndarray:
        """
        Compute a vein mask at the given radius threshold.

        Used for both the permissive mask (temporary artery tracing boundary,
        step 3) and the final mask (definitive vein labels, step 5).

        Parameters
        ----------
        restrictive_veins : np.ndarray
            High-confidence vein mask (always included in result).
        artery : np.ndarray
            Current artery labels (excluded from result).
        radius_threshold : float
            Minimum radius for an edge to qualify as a vein candidate.
            ``permissive_vein_radius`` for tracing boundary,
            ``final_vein_radius`` for definitive labels.
        """
        s   = self.signals
        cfg = self.cfg

        large = s.radii >= radius_threshold
        seed  = restrictive_veins | large

        if s.vein_intensity is not None:
            seed = seed | (s.vein_intensity >= cfg.vein_intensity_min)
        if s.vein_binary is not None:
            seed = seed | s.vein_binary.astype(bool)

        return seed & ~artery

    def _trace_arteries(self, veins: np.ndarray) -> None:
        """
        Hysteresis extension of artery labels.

        Walks along edges that are:
        - large enough (``>= artery_trace_radius``)
        - have sufficient artery marker intensity
        - are not veins
        - are not too close to the brain surface

        Updates ``graph['artery']`` in place.
        """
        s   = self.signals
        cfg = self.cfg

        artery = self.graph.edge_property('artery')

        condition_args = {
            'distance_to_surface': s.distance_to_surface,
            'distance_threshold':  cfg.distance_to_surface_min,
            'vein':                veins,
            'radii':               s.radii,
            'artery_trace_radius': cfg.artery_trace_radius,
            'artery_intensity':    s.artery_intensity,
            'artery_intensity_min': cfg.artery_intensity_min,
        }

        def continue_edge(graph, edge, **kw):
            if kw['distance_to_surface'][edge] < kw['distance_threshold']:
                return False
            if kw['vein'][edge]:
                return False
            return (kw['radii'][edge] >= kw['artery_trace_radius'] and
                    kw['artery_intensity'][edge] >= kw['artery_intensity_min'])

        traced = graph_processing.trace_edge_label(
            self.graph, artery,
            condition=continue_edge,
            max_iterations=cfg.max_artery_trace_iter,
            **condition_args)

        self.graph.define_edge_property('artery', traced)

    def _trace_veins(self) -> None:
        """
        Hysteresis extension of vein labels.

        Walks along edges that are:
        - large enough (``>= vein_trace_radius``)
        - not too close to an artery (1-edge buffer)
        - (optionally) have sufficient vein marker intensity
        - (optionally) have low artery marker intensity

        Updates ``graph['vein']`` in place.
        """
        s   = self.signals
        cfg = self.cfg

        artery = self.graph.edge_property('artery')
        min_distance_to_artery = 1
        artery_expanded = self.graph.edge_dilate_binary(artery, steps=min_distance_to_artery)

        condition_args = {
            'artery_expanded':   artery_expanded,
            'radii':             s.radii,
            'vein_trace_radius': cfg.vein_trace_radius,
            'vein_intensity':    s.vein_intensity,
            'vein_intensity_min': cfg.vein_intensity_min,
            'artery_intensity':  s.artery_intensity,
            'vein_intensity_range': cfg.vein_intensity_range,
        }

        def continue_edge(graph, edge, **kw):
            if kw['artery_expanded'][edge]:
                return False
            if kw['radii'][edge] < kw['vein_trace_radius']:
                return False
            if kw['vein_intensity'] is not None:
                if kw['vein_intensity'][edge] < kw['vein_intensity_min']:
                    return False
            if kw['artery_intensity'] is not None:
                lo, hi = kw['vein_intensity_range']
                if not (lo <= kw['artery_intensity'][edge] <= hi):
                    return False
            return True

        traced = graph_processing.trace_edge_label(
            self.graph, self.graph.edge_property('vein'),
            condition=continue_edge,
            max_iterations=cfg.max_vein_trace_iter,
            **condition_args)

        self.graph.define_edge_property('vein', traced)

    def _remove_small_components(self, property_name: str,
                                  min_edges: int) -> None:
        """
        Remove connected components of ``property_name`` with fewer
        than ``min_edges`` edges.  Reclassified as capillaries.
        """
        label = self.graph.edge_property(property_name)

        sub = self.graph.sub_graph(edge_filter=label, view=True)
        sub_edge, edge_map = sub.edge_graph(return_edge_map=True)
        components, sizes = sub_edge.label_components(return_vertex_counts=True)

        remove = edge_map[np.isin(components, np.where(sizes < min_edges)[0])]
        label[remove] = False

        self.graph.define_edge_property(property_name, label)