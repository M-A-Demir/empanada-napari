from abc import ABC, abstractmethod

import numpy as np

from .union_find import UnionFind
from .result import ArrayResult, TrackerResult


class ChunkReconciler(ABC):
    r"""The second Bridge axis (report-2026-07-16-segmentation-architecture.md):
    how cross-chunk object identity is resolved differs by representation
    (dense array vs InstanceTracker), not by which InferenceStrategy
    produced a chunk's result -- the same hooks cover both, so
    ChunkedExecutor.run_workflow stays identical in shape for every
    strategy that declares a reconciler_cls.

    Hooks, in the order ChunkedExecutor calls them for one chunk:
      1. uniquify         -- relabel this chunk's result into its own
                              reserved id window (self.class_ids[uid]),
                              so ids never collide across chunks.
      2. checkpoint        -- persist the (now-unique) result and return
                              a lightweight handle for later lookups.
      3. record_matches    -- (skipped for panels, which have no earlier
                              neighbours) compare against already-
                              finalized neighbours and record every
                              match above threshold as an edge. Never
                              mutates anything directly.
    Then, once every wave has run:
      4. apply_final_remap -- resolve every recorded edge into one
                              old_id -> new_id map via a single, global
                              union-find, and apply it everywhere.
      5. assemble_result    -- produce the SegmentationResult that the
                              strategy's finalize() expects.

    Recording every match above threshold as an edge (rather than
    immediately relabeling to the single best match) is what lets a
    multi-way merge collapse correctly: an object a boundary strip
    reveals is the same as two separately-labeled panel fragments needs
    all three ids in one connected component, not one fragment adopted
    and the other silently orphaned (report-2026-07-15-orthoplane-strategy.md,
    section 4).
    """

    def __init__(self, match_conf_thr=0.8):
        # Two genuinely distinct objects have disjoint pixel sets, so for
        # any single candidate, conf_a + conf_b <= 1 against two different
        # already-established objects. A threshold above 0.5 is what makes
        # it arithmetically impossible for one candidate to ever match two
        # different objects at once -- the guarantee that stops
        # record_matches from merging two objects an earlier chunk
        # correctly kept separate (e.g. touching organelles a panel
        # already told apart) just because one later, ambiguous candidate
        # happens to overlap both. Lowering this below 0.5 would silently
        # reintroduce that failure mode.
        assert match_conf_thr > 0.5, 'match_conf_thr must be > 0.5, see comment above'
        self.match_conf_thr = match_conf_thr
        self._edges = []
        self._uf = UnionFind()

    @abstractmethod
    def uniquify(self, result, uid, class_ids):
        r"""Relabels one chunk's result into its reserved id window
        (class_ids[uid]) and returns the (mutated) result."""

    @abstractmethod
    def checkpoint(self, result, slice_idx, uid, context):
        r"""Persists one chunk's (already-uniquified) result and returns
        a handle that gather_neighbours/record_matches/apply_final_remap
        can use to look it up again. `context` carries whatever
        representation-specific backing store or metadata this
        reconciler type needs (e.g. a zarr store for arrays, the global
        volume shape for trackers)."""

    def gather_neighbours(self, slice_idx, finalized):
        r"""Read-only neighbour lookup, representation-agnostic: since
        `finalized` only ever contains chunk types that have already
        fully completed (panels, then earlier strip waves, in the strict
        wave order run_workflow enforces), a chunk reconciles against the
        right set just by checking spatial overlap against everything
        finalized so far -- no separate "which types come before me"
        bookkeeping needed.
        """
        neighbours = []
        for chunks in finalized.values():
            for other_slice_idx, handle in chunks.values():
                if self._chunks_overlap(slice_idx, other_slice_idx):
                    neighbours.append(handle)
        return neighbours

    @abstractmethod
    def record_matches(self, candidate_handle, neighbour_handles, context):
        r"""Finds every already-finalized neighbour the candidate matches
        above self.match_conf_thr and records each as an edge (via
        self._record_edges). Never mutates candidate_handle or any
        neighbour directly -- collapsing ids is deferred to a single
        global union-find in apply_final_remap, so multi-way merges
        collapse correctly regardless of the order chunks are discovered
        in."""

    @abstractmethod
    def apply_final_remap(self, finalized, context):
        r"""Resolves every edge recorded across every wave into one
        old_id -> new_id map and applies it to every finalized chunk."""

    @abstractmethod
    def assemble_result(self, finalized, context):
        r"""Produces the SegmentationResult that the owning strategy's
        finalize() expects, from every chunk's (already remapped)
        contribution."""

    def _chunks_overlap(self, slice_a, slice_b):
        r"""Two chunks overlap iff their bounds intersect along every
        dimension -- covers both plain panel/strip borders and the
        strip/strip crossing overlaps described in
        report-2026-07-15-orthoplane-strategy.md section 2, with no
        special-casing needed."""
        return all(
            a.start < b.stop and b.start < a.stop
            for a, b in zip(slice_a, slice_b)
        )

    def _record_edges(self, edges):
        # Reconciler instances are shared across a wave's parallel jobs
        # (ChunkedExecutor uses joblib's threading backend specifically
        # so the torch engine and any shared zarr store don't need to be
        # re-pickled per worker -- see chunked.py). list.append is atomic
        # under CPython's GIL, so concurrent calls from multiple threads
        # are safe without extra locking.
        self._edges.extend(edges)

    def _build_id_map(self):
        for a, b in self._edges:
            self._uf.union(a, b)
        return {
            node: self._uf.find(node)
            for edge in self._edges for node in edge
        }


class ArrayChunkReconciler(ChunkReconciler):
    r"""Array-representation reconciliation, for SingleSliceStrategy: each
    chunk checkpoints its labeled array straight to a shared zarr store
    (the array is memory-heavy, so persisting happens every wave, not
    just at the end); matches are found by re-reading already-written
    neighbouring regions back from disk; the final remap is itself
    applied chunk-by-chunk, since the whole array is too large to hold
    in memory at once.

    Match algorithm -- flagged for your review: the original array-based
    reconciliation (build_mapping in the pre-refactor parallel.py) gave
    each candidate label its single best-matching neighbour and relabeled
    immediately. That silently breaks when a boundary-strip object
    bridges two already-labeled panel fragments of the same real object:
    it needs all 3 ids collapsed into 1, not one fragment adopted and the
    other orphaned. This class instead reuses the same record-all-
    matches-then-global-union-find approach designed for the tracker path
    (report-2026-07-15-orthoplane-strategy.md, section 4) for parity
    across both reconciler types.

    This was an explicit but *unconfirmed* design decision at the time
    this was written -- if it over-merges, under-performs, or otherwise
    doesn't behave as wanted on real chunked 2D data, the documented
    fallback is to restore the original single-best-match `build_mapping`
    behaviour here instead (immediately relabel each candidate to its one
    best match, no union-find, no edge deferral) -- see git history for
    empanada/seg_executors/parallel.py.
    """

    def uniquify(self, result, uid, class_ids):
        seg = result.seg
        unique_labels = np.unique(seg[seg > 0])
        if len(unique_labels) == 0:
            return result

        next_id = class_ids[uid]
        max_key = int(unique_labels.max())
        lut = np.arange(max_key + 1, dtype=np.int64)
        for i, old_label in enumerate(unique_labels):
            lut[old_label] = next_id + i

        mask = seg > 0
        seg = seg.copy()
        seg[mask] = lut[seg[mask]]
        result.seg = seg
        return result

    def gather_neighbours(self, slice_idx, finalized):
        # Deliberately does NOT use the base class's slice-overlap search
        # across other chunks' separate slice_idx values (that's needed
        # for the tracker path, where each chunk's data is a separate
        # in-memory object with its own local coordinate frame). Here,
        # every chunk's data already lives in ONE shared, global-
        # coordinate zarr_store -- whatever an earlier panel or wave
        # wrote into this chunk's own overlapping slice_idx region is
        # already directly readable by re-reading that same slice_idx,
        # with no cross-chunk coordinate translation needed at all. See
        # checkpoint(), which does exactly that.
        return []

    def checkpoint(self, result, slice_idx, uid, context):
        r"""Reads whatever earlier panels/waves already wrote into this
        chunk's own slice_idx region of the shared store (a strip's
        slice_idx intentionally overlaps its bordering panels' regions --
        report-2026-07-15-orthoplane-strategy.md, section 2), matches
        this chunk's own labels against that existing data (recording
        edges, never relabeling immediately -- see class docstring), then
        merge-writes: only pixels where this chunk found foreground are
        overwritten, so pixels the existing data already covers (but this
        chunk's local inference didn't re-detect) are left untouched
        rather than being clobbered with background.

        This is also why record_matches is a no-op below: unlike the
        tracker path, where comparing a candidate against a neighbour
        requires them to first be pulled into the same coordinate frame,
        an array chunk's "neighbours" are just whatever's already sitting
        in the same shared, global-coordinate store -- matching has to
        happen right here, before this chunk's own write, or the data it
        would need to compare against is gone.
        """
        zarr_store = context['zarr_store']
        seg = result.seg
        existing = zarr_store[slice_idx]

        if np.any(existing > 0):
            cand_labels = np.unique(seg[seg > 0])
            edges = []
            for cand_label in cand_labels:
                mask = seg == cand_label
                overlap = existing[mask]
                overlap = overlap[overlap > 0]
                if len(overlap) == 0:
                    continue

                labels, counts = np.unique(overlap, return_counts=True)
                best = np.argmax(counts)
                conf = counts[best] / counts.sum()
                if conf >= self.match_conf_thr:
                    edges.append((int(cand_label), int(labels[best])))

            self._record_edges(edges)

        merged = existing.copy()
        mask = seg > 0
        merged[mask] = seg[mask]
        zarr_store[slice_idx] = merged

        # the handle IS the slice_idx: the labeled data lives in
        # zarr_store, looked up by slice_idx, whenever it's next needed
        return slice_idx

    def record_matches(self, candidate_handle, neighbour_handles, context):
        # No-op: matching against already-finalized neighbours happens
        # inside checkpoint() above, since for this representation
        # "reading the neighbours" and "persisting this chunk's own
        # result" are the same physical read-then-merge-write operation,
        # not two separable steps.
        return

    def apply_final_remap(self, finalized, context):
        zarr_store = context['zarr_store']
        id_map = self._build_id_map()
        if not id_map:
            return

        max_key = max(id_map)
        lut = np.arange(max_key + 1, dtype=np.int64)
        for old, new in id_map.items():
            lut[old] = new

        for chunks in finalized.values():
            for slice_idx, _ in chunks.values():
                chunk = zarr_store[slice_idx]
                mask = (chunk > 0) & (chunk <= max_key)
                chunk[mask] = lut[chunk[mask]]
                zarr_store[slice_idx] = chunk

    def assemble_result(self, finalized, context):
        # Chunks write to spatially disjoint regions of the shared zarr
        # store, so giving two fragments the same final id (via
        # apply_final_remap) is enough -- the union happens "for free"
        # once everything is on disk. No separate assembly step needed,
        # unlike the tracker path (report section 2).
        return ArrayResult(context['zarr_store'])


class TrackerChunkReconciler(ChunkReconciler):
    r"""Tracker-representation reconciliation, for StackStrategy and
    OrthoplaneStrategy: trackers stay the source of truth end-to-end
    (report-2026-07-15-orthoplane-strategy.md, section 1) -- nothing is
    written to disk until the single, final rasterization pass, so
    checkpointing here just means holding the (uniquified,
    coordinate-translated) tracker in memory. Matching uses rle_ioa
    (candidate's own footprint as the denominator) at the same 0.8
    default as the original array-based reconciliation.

    Ported closely from example_workflow_sketch.py (v2), which worked
    through the coordinate-frame and multi-way-merge issues in detail
    before this was written.
    """

    def uniquify(self, result, uid, class_ids):
        base_id = class_ids[uid]
        for trackers in result.trackers_dict.values():
            for tracker in trackers:
                next_id = base_id
                remapped = {}
                for attrs in tracker.instances.values():
                    remapped[next_id] = attrs
                    next_id += 1
                tracker.instances = remapped
        return result

    def _translate_to_global(self, tracker, slice_idx, global_shape):
        r"""Converts a chunk-local tracker's RLE coordinates into flat
        indices relative to the full (downsampled) volume.

        InstanceTracker.update() computes starts/runs as flat indices
        relative to self.shape3d -- the volume actually passed to
        infer_on_axis, which for a chunk is the chunk's own local shape,
        not the full volume's. Without translating first, two chunks'
        RLEs with numerically identical values would refer to completely
        different physical voxels, making both matching and whole-volume
        assembly silently wrong rather than loudly broken. This is the
        least-validated, most correctness-critical piece of this class --
        it has not been checked against real chunked volume data, only
        reasoned through (see report-2026-07-15-orthoplane-strategy.md,
        Open Questions).

        A plain index shift is not enough: a run that's contiguous in a
        narrower local chunk width won't stay contiguous once
        reinterpreted against the full volume's width, since ravel/
        unravel strides differ between the two shapes. Full decode ->
        translate -> re-encode is required.
        """
        from empanada.array_utils import rle_decode, rle_encode

        offset = np.array([s.start for s in slice_idx])
        local_shape = tracker.shape3d
        ndim = len(offset)

        for attrs in tracker.instances.values():
            flat_local = rle_decode(attrs['starts'], attrs['runs'])
            local_coords = np.unravel_index(flat_local, local_shape)
            global_coords = tuple(c + o for c, o in zip(local_coords, offset))
            flat_global = np.ravel_multi_index(global_coords, global_shape)

            starts, runs = rle_encode(np.sort(flat_global, kind='stable'))
            attrs['starts'], attrs['runs'] = starts, runs

            box = attrs['box']
            attrs['box'] = tuple(box[i] + offset[i % ndim] for i in range(len(box)))

        tracker.shape3d = global_shape
        return tracker

    def checkpoint(self, result, slice_idx, uid, context):
        global_shape = context['global_shape']
        handle = {}
        for axis_name, trackers in result.trackers_dict.items():
            for tracker in trackers:
                translated = self._translate_to_global(tracker, slice_idx, global_shape)
                handle[(axis_name, translated.class_id)] = translated
        return handle

    def record_matches(self, candidate_handle, neighbour_handles, context):
        from empanada.array_utils import rle_ioa

        neighbours_by_group = {}
        for handle in neighbour_handles:
            for group, tracker in handle.items():
                neighbours_by_group.setdefault(group, []).append(tracker)

        edges = []
        for group, candidate_tracker in candidate_handle.items():
            for ref_tracker in neighbours_by_group.get(group, []):
                for cand_id, cand_attrs in candidate_tracker.instances.items():
                    for ref_id, ref_attrs in ref_tracker.instances.items():
                        # rle_ioa's denominator is its SECOND argument
                        # (area = runs_b.sum()), so the candidate must be
                        # passed second to get intersection-over-the-
                        # candidate's-own-area, matching the original
                        # array-based reconciliation's convention exactly.
                        conf = rle_ioa(
                            ref_attrs['starts'], ref_attrs['runs'],
                            cand_attrs['starts'], cand_attrs['runs'],
                        )
                        if conf >= self.match_conf_thr:
                            edges.append((cand_id, ref_id))

        self._record_edges(edges)

    def apply_final_remap(self, finalized, context):
        id_map = self._build_id_map()
        for chunks in finalized.values():
            for _, handle in chunks.values():
                for tracker in handle.values():
                    tracker.instances = {
                        id_map.get(old_id, old_id): attrs
                        for old_id, attrs in tracker.instances.items()
                    }

    def assemble_result(self, finalized, context):
        r"""Merges every chunk's (already remapped, already globally-
        coordinated) per-(axis, class) tracker into one tracker spanning
        the whole volume -- equivalent to what engine.infer_on_axis would
        have produced directly on the full array. Chunks that now share a
        final id (an object completed or bridged via a strip) have their
        RLE fragments concatenated, not dict.update()'d, or all but one
        chunk's contribution would silently vanish -- mirrors what
        InstanceTracker.update()/.finish() already do when accumulating
        multiple 2D slices for one instance during normal, non-chunked
        inference."""
        from empanada.array_utils import merge_boxes
        from empanada.inference.tracker import InstanceTracker

        global_trackers = {}  # {(axis_name, class_id): InstanceTracker}
        for chunks in finalized.values():
            for _, handle in chunks.values():
                for group, tracker in handle.items():
                    axis_name, class_id = group
                    if group not in global_trackers:
                        global_trackers[group] = InstanceTracker(
                            class_id, tracker.label_divisor, tracker.shape3d, axis_name
                        )

                    target = global_trackers[group].instances
                    for instance_id, attrs in tracker.instances.items():
                        if instance_id not in target:
                            target[instance_id] = {
                                'box': attrs['box'],
                                'starts': [attrs['starts']],
                                'runs': [attrs['runs']],
                            }
                        else:
                            existing = target[instance_id]
                            existing['box'] = merge_boxes(existing['box'], attrs['box'])
                            existing['starts'].append(attrs['starts'])
                            existing['runs'].append(attrs['runs'])

        for tracker in global_trackers.values():
            tracker.finish()

        trackers_dict = {}
        for (axis_name, class_id), tracker in global_trackers.items():
            trackers_dict.setdefault(axis_name, []).append(tracker)

        return TrackerResult(trackers_dict)
