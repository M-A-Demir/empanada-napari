from abc import ABC, abstractmethod


class SegmentationResult(ABC):
    r"""Wraps whatever an InferenceStrategy produces so that an Executor,
    and later a ChunkReconciler, can move a result around (checkpoint it,
    reconcile it against neighbours, hand it back to the caller) without
    caring whether the underlying representation is a dense labeled array
    (2D strategies) or a set of InstanceTrackers (3D strategies).

    This is the seam the two representations meet at: everything upstream
    of it (InferenceStrategy.run) and downstream of it (ChunkReconciler,
    Executor.run_workflow's return value) only ever needs to know that a
    SegmentationResult exists, never which concrete subclass it is.
    """

    @abstractmethod
    def to_return_value(self):
        r"""Converts back to whatever shape callers of run_workflow
        (SliceSegPipeline/VolumeSegPipeline) historically expect."""


class ArrayResult(SegmentationResult):
    r"""Wraps a dense labeled array produced by an array-based strategy
    (SingleSliceStrategy, BatchSliceStrategy). axis/plane/y/x carry the
    same positional metadata the pre-refactor `_run_model`/`_run_model_batch`
    returned alongside the segmentation, used by the GUI layer to place
    the result back into the viewer at the right offset.
    """

    def __init__(self, seg, axis=None, plane=None, y=None, x=None):
        self.seg = seg
        self.axis = axis
        self.plane = plane
        self.y = y
        self.x = x

    def to_return_value(self):
        return self.seg, self.axis, self.plane, self.y, self.x


class TrackerResult(SegmentationResult):
    r"""Wraps the InstanceTracker output produced by a tracker-based
    strategy (StackStrategy, OrthoplaneStrategy).

    trackers_dict: {axis_name: [InstanceTracker, ...]} -- one list of
        per-class trackers per axis that was actually run. A stack
        (single-axis) result has exactly one key; an orthoplane result
        has 'xy', 'xz', 'yz'.
    axes_dict: {axis_name: stack_or_None} -- the rasterized panoptic
        stack for each axis, if the engine was configured to produce one
        (Engine3d.save_panoptic), else None per axis.
    """

    def __init__(self, trackers_dict, axes_dict=None):
        self.trackers_dict = trackers_dict
        self.axes_dict = axes_dict or {}

    def to_return_value(self):
        return self.trackers_dict, self.axes_dict
