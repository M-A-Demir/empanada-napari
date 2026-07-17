from .executor import Executor


class ChunkedExecutor(Executor):
    r"""Runs the strategy on N chunks (an initial panel split, then
    boundary-strip waves) of a large image/volume, reconciling
    cross-chunk object identity via a ChunkReconciler -- chosen once, by
    the strategy itself, via strategy.reconciler_cls (see strategy.py) --
    then calls finalize() exactly once, globally, after reconciliation,
    never per chunk.

    Renamed from ParallelExecutor (report-2026-07-16-segmentation-architecture.md):
    "parallel" described concurrency, which turned out to be orthogonal
    to what this class actually does -- you could chunk without running
    chunks concurrently and still need the same reconciliation machinery.
    See SingleRegionExecutor's docstring for the equivalent point there.

    NOTE: run_workflow is completed in a later branch alongside
    ChunkReconciler (report-2026-07-15-orthoplane-strategy.md section 6);
    this branch establishes the class shape and the chunk id
    partitioning scheme, which is representation-agnostic and already
    validated (reused as-is from the pre-refactor parallel.py /
    example_workflow.py, per that report's section 3).
    """

    def __init__(self, strategy, zarr_inpath=None, zarr_outpath=None, scale=2):
        super().__init__(strategy)
        if strategy.reconciler_cls is None:
            raise ValueError(
                f'{type(strategy).__name__} declares no reconciler_cls, so it '
                'has no way to resolve object identity across chunk boundaries '
                'and cannot be run inside a ChunkedExecutor. Use a '
                'SingleRegionExecutor instead.'
            )
        self.zarr_inpath = zarr_inpath
        self.zarr_outpath = zarr_outpath
        self.scale = scale
        self.padding = 200
        self.class_ids = {}

    # ---------------- chunk identity / id space --------------------
    # Unchanged from example_workflow.py / the pre-refactor parallel.py:
    # each chunk gets a stable, disjoint id window, computed once,
    # single-threaded, before a wave's parallel dispatch -- so parallel
    # workers only ever read self.class_ids, never mutate it, sidestepping
    # any shared-counter race condition entirely.
    def _get_chunk_uid(self, slice_idx):
        coords = [(s.start if i < len(slice_idx) - 1 else s.stop - 1)
                  for i, s in enumerate(slice_idx)]
        return "_".join(map(str, coords))

    def _create_class_ids(self, chunk_indices):
        divisor = 1000
        class_id = max(self.class_ids.values()) + 1 if self.class_ids else 1

        for slice_idx in chunk_indices:
            uid = self._get_chunk_uid(slice_idx)
            self.class_ids[uid] = class_id * divisor
            class_id += 1

    def run_workflow(self, engine, image, **kwargs):
        raise NotImplementedError(
            'ChunkedExecutor.run_workflow is completed in the '
            'feature/chunk-reconciler branch, alongside ChunkReconciler.'
        )
