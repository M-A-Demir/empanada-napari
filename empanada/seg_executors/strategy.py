from abc import ABC, abstractmethod

import numpy as np
import dask.array as da
from tqdm import tqdm

from .result import ArrayResult, TrackerResult
from .reconciler import ArrayChunkReconciler, TrackerChunkReconciler


class InferenceStrategy(ABC):
    r"""The algorithm axis of the Executor x InferenceStrategy bridge
    (report-2026-07-16-segmentation-architecture.md): exactly what runs
    on the pixels for one region (a single slice, a batch of independent
    slices, a single-axis stack, or all three orthoplane axes) -- never
    how many regions it's asked to run on, or how work across them is
    distributed. That's Executor's concern; this class never references
    an Executor and knows nothing about chunking.

    reconciler_cls fixes, once per subclass, which ChunkReconciler
    resolves cross-chunk object identity when this strategy is run
    inside a ChunkedExecutor. Declaring it here -- rather than branching
    on array-vs-tracker somewhere inside ChunkedExecutor -- keeps that
    decision in exactly one place, consistent with strategy selection
    itself already being centralized wherever a Pipeline picks a
    strategy. A value of None means this strategy can only ever run
    inside a SingleRegionExecutor (see ChunkedExecutor.__init__, which
    enforces this at construction time).
    """

    #: ChunkReconciler subclass to use when this strategy is run inside
    #: a ChunkedExecutor. None means this strategy can only ever run
    #: inside a SingleRegionExecutor.
    reconciler_cls = None

    def __init__(self, fill_holes_in_segmentation=False):
        self.fill_holes = fill_holes_in_segmentation

    @abstractmethod
    def run(self, engine, image, **kwargs):
        r"""Runs inference on one region and returns a SegmentationResult."""

    def finalize(self, result, **kwargs):
        r"""Called exactly once, after any reconciliation, regardless of
        executor type -- trivially, immediately after run() for a
        SingleRegionExecutor, or after global reconciliation for a
        ChunkedExecutor. Default is identity: only OrthoplaneStrategy
        overrides this, to compute consensus, since that computation
        must happen exactly once, globally, and specifically must never
        run per-chunk. This is what keeps "no per-chunk consensus" a
        structural guarantee rather than a special case Executor has to
        know about.
        """
        return result

    def _fill_holes_in_segmentation(self, mask):        
        """Shared by the array-based strategies. Moved here (off the
        old shared Executor base class) so tracker-based strategies
        never inherit a method that only works on dense 2D label arrays
        -- see bugfix/executor-known-bugs for the guard this replaces.
        """
        unique_indices = np.unique(mask)
        rprops = measure.regionprops(mask)

        # crop labels and then apply fill holes
        for rp in tqdm(rprops, desc='filling holes in labels:'):
            if rp.label in unique_indices and rp.label > 0:
                minr, minc, maxr, maxc = rp.bbox

                tmp = mask[minr:maxr, minc:maxc]
                tmp = binary_fill_holes(tmp.astype(bool))
                mask[minr:maxr, minc:maxc] = tmp.astype(mask.dtype) * rp.label
        return mask


class SingleSliceStrategy(InferenceStrategy):
    r"""2D, non-batch: engine.infer() on one image, dense array output.
    Moved near-unchanged from Executor._run_model. This is the strategy
    the zarr_2dinference_implement branch's chunked (panel + boundary
    strip) 2D workflow is built around, hence ArrayChunkReconciler."""

    reconciler_cls = ArrayChunkReconciler

    def run(self, engine, image, axis=None, plane=None, y=None, x=None, **kwargs):
        if isinstance(image, da.core.Array):
            image = image.compute()
        seg = engine.infer(image)
        if self.fill_holes:
            seg = self._fill_holes(seg)
        return ArrayResult(seg, axis, plane, y, x)


class BatchSliceStrategy(InferenceStrategy):
    r"""2D batch mode: each slice of a stack is segmented independently.
    Moved near-unchanged from Executor._run_model_batch. Deliberately
    has no reconciler_cls: batch mode slices are independent frames of a
    stack, not spatial chunks of one image, so there is never any
    cross-chunk identity to resolve here -- this matches the original
    behaviour, where batch mode never reconciled labels across slices.
    """

    def run(self, engine, image, **kwargs):
        if image.ndim == 3:
            print(f'Running batch mode inference on {len(image)} images.')
            segmentations = []
            for img_slice in tqdm(image, total=len(image)):
                if isinstance(img_slice, da.core.Array):
                    img_slice = img_slice.compute()

                seg = engine.infer(img_slice)
                if self.fill_holes:
                    seg = self._fill_holes(seg)
                segmentations.append(seg)

            # stack segmentations with padding
            max_h = max(seg.shape[0] for seg in segmentations)
            max_w = max(seg.shape[1] for seg in segmentations)
            padded = []
            for seg in segmentations:
                h, w = seg.shape
                padh, padw = max_h - h, max_w - w
                padded.append(np.pad(seg, ((0, padh), (0, padw))))

            return ArrayResult(np.stack(padded, axis=0))

        elif image.ndim == 2:
            if isinstance(image, da.core.Array):
                image = image.compute()

            seg = engine.infer(image)
            if self.fill_holes:
                seg = self._fill_holes(seg)
            return ArrayResult(seg)

        else:
            raise Exception(f'Batch mode supports 2d and 3d, got {image.ndim}d.')


class StackStrategy(InferenceStrategy):
    r"""3D, single-axis: engine.infer_on_axis() on one volume, tracker
    output for exactly that axis. Moved near-unchanged from
    Executor._stack_inference."""

    reconciler_cls = TrackerChunkReconciler

    def run(self, engine, volume, axis_name='xy', **kwargs):
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        return TrackerResult({axis_name: trackers}, {axis_name: stack})


class OrthoplaneStrategy(InferenceStrategy):
    r"""3D orthoplane: engine.infer_on_axis() run once per axis (xy, xz,
    yz), tracker output for all three. Moved near-unchanged from
    Executor._orthoplane_inference, with consensus split out into
    finalize() (see report-2026-07-16-segmentation-architecture.md,
    "Resolved whether 'parallel = N serial calls' holds"): orthoplane
    consensus must run exactly once, globally, never per chunk, which is
    exactly what makes this the one strategy whose finalize() isn't the
    default identity.
    """

    reconciler_cls = TrackerChunkReconciler

    def run(self, engine, volume, **kwargs):
        trackers_dict = {}
        axes_dict = {}
        for axis_name in ['xy', 'xz', 'yz']:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            trackers_dict[axis_name] = trackers

            # report instances per class
            for tracker in trackers:
                class_id = tracker.class_id
                print(f'Class {class_id}, axis {axis_name}, has {len(tracker.instances.keys())} instances')
            axes_dict[axis_name] = stack

        return TrackerResult(trackers_dict, axes_dict)

    def finalize(self, result, engine=None, pixel_vote_thr=2,
                 cluster_iou_thr=0.75, allow_one_view=False, **kwargs):
        from empanada.inference.patterns import (
            get_axis_trackers_by_class,
            create_instance_consensus,
            create_semantic_consensus,
        )

        consensus_trackers = {}
        for class_id in engine.labels:
            class_trackers = get_axis_trackers_by_class(result.trackers_dict, class_id)
            if class_id in engine.thing_list:
                consensus_trackers[class_id] = create_instance_consensus(
                    class_trackers, pixel_vote_thr, cluster_iou_thr, allow_one_view
                )
            else:
                consensus_trackers[class_id] = create_semantic_consensus(
                    class_trackers, pixel_vote_thr
                )

        return TrackerResult({'consensus': list(consensus_trackers.values())}, result.axes_dict)
