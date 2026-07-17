import joblib
import dask.array as da

from .executor import Executor
from .reconciler import ArrayChunkReconciler
from empanada.zarr_utils import _write_empty_chunk, _generate_tiles


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

    Execution order (report-2026-07-15-orthoplane-strategy.md, section 6):
    panels (parallel, no deps) -> x_strip wave -> y_strip wave -> z_strip
    wave [3D only] -> global id remap (single union-find over every edge
    recorded across every wave) -> assemble the reconciled result ->
    strategy.finalize() (identity for everything except OrthoplaneStrategy,
    which computes consensus here). Each wave must fully complete before
    the next starts, since neighbour lookups in that wave need the
    previous wave's chunks already sitting in `finalized`.
    """

    def __init__(self, strategy, zarr_inpath=None, zarr_outpath=None,
                 scale=2, match_conf_thr=0.8, multiscale_writer=None):
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
        self.reconciler = strategy.reconciler_cls(match_conf_thr=match_conf_thr)
        # Optional hook so a Pipeline (which already knows how to read
        # the *input* zarr's OME metadata) can supply its own multiscale
        # pyramid writer. Kept as an injected callable rather than a
        # method this class implements itself, since that metadata-
        # reading logic belongs to the caller, not to chunk reconciliation
        # -- see SliceSegPipeline/VolumeSegPipeline.write_out_multiscale.
        self.multiscale_writer = multiscale_writer

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

    def _downsample_array(self, image, scale):
        # Generalized from the pre-refactor parallel.py, which always
        # downsampled exactly 2 axes (image[::scale, ::scale]) regardless
        # of image.ndim -- correct for the 2D case this class was
        # originally built around, but silently wrong for a 3D volume,
        # where the trailing axis would never be downsampled at all.
        return image[tuple(slice(None, None, scale) for _ in range(image.ndim))]

    def _get_strips(self, shape):
        r"""Boundary strips around each internal chunk-grid line, keyed
        by direction name ('x', 'y', and 'z' for a 3D shape) rather than
        the positional axis index the pre-refactor parallel.py used --
        that mismatch meant run_workflow's `strips_by_direction.get(direction,
        [])` lookup could never actually find anything under the old
        implementation, since strips were keyed 0/1/2 there, not 'x'/'y'/'z'.

        Direction names are assigned from the END of the shape tuple (x
        is always last, y second-to-last, z third-to-last if present),
        so this works uniformly whether `shape` is a 2D (y, x) downsampled
        image or a 3D (z, y, x) downsampled volume.
        """
        ndim = len(shape)
        dir_names = ['z', 'y', 'x'][-ndim:]
        tile_shape = [dim // self.scale for dim in shape]

        edges = []
        for dim, step in enumerate(tile_shape):
            s = shape[dim]
            e = list(range(0, s, step))
            if e[-1] != s:
                e.append(s)
            edges.append(e)

        strips = {name: [] for name in dir_names}
        for axis, name in enumerate(dir_names):
            for boundary in edges[axis][1:-1]:
                slc = []
                for d in range(ndim):
                    if d == axis:
                        slc.append(slice(
                            max(0, boundary - self.padding),
                            min(shape[d], boundary + self.padding)
                        ))
                    else:
                        slc.append(slice(0, shape[d]))
                strips[name].append(tuple(slc))

        return strips

    # ---------------- one chunk, end to end -------------------------
    def _process_chunk(self, engine, image, slice_idx, chunk_type,
                        neighbour_handles, context, **kwargs):
        uid = self._get_chunk_uid(slice_idx)
        chunk_image = image[slice_idx]
        if isinstance(chunk_image, da.Array):
            chunk_image = chunk_image.compute()

        result = self.strategy.run(engine, chunk_image, **kwargs)
        result = self.reconciler.uniquify(result, uid, self.class_ids)
        handle = self.reconciler.checkpoint(result, slice_idx, uid, context)

        if chunk_type != 'panel':  # panels have no earlier-finalized neighbours
            self.reconciler.record_matches(handle, neighbour_handles, context)

        return uid, slice_idx, handle

    def _run_wave(self, engine, image, indices, chunk_type,
                  neighbours_by_idx, context, **kwargs):
        # threading, not joblib's process-based default: workers share
        # the torch engine and (for array strategies) the zarr store
        # directly, so nothing needs to be re-pickled per worker, and
        # self.reconciler can accumulate edges directly (see
        # ChunkReconciler._record_edges) instead of returning them for
        # a driver to fold in afterward.
        jobs = joblib.Parallel(n_jobs=-1, prefer='threads')(
            joblib.delayed(self._process_chunk)(
                engine, image, slice_idx, chunk_type,
                neighbours_by_idx.get(slice_idx, []), context, **kwargs
            )
            for slice_idx in indices
        )

        return {uid: (slice_idx, handle) for uid, slice_idx, handle in jobs}

    def run_workflow(self, engine, image, **kwargs):
        image = self._downsample_array(image, self.scale)
        global_shape = image.shape
        tile_shape = tuple(max(1, dim // self.scale) for dim in image.shape)
        panel_indices = list(_generate_tiles(image.shape, tile_shape))

        context = {'global_shape': global_shape}
        is_array_based = isinstance(self.reconciler, ArrayChunkReconciler)
        if is_array_based:
            context['zarr_store'] = _write_empty_chunk(
                self.zarr_outpath, image, inp_scale=[0.005, 0.005]
            )

        self._create_class_ids(panel_indices)
        finalized = {'panel': self._run_wave(
            engine, image, panel_indices, 'panel', {}, context, **kwargs
        )}

        strips_by_direction = self._get_strips(image.shape)
        for chunk_type in ('x_strip', 'y_strip', 'z_strip'):
            direction = chunk_type.split('_')[0]
            strip_indices = strips_by_direction.get(direction, [])
            if not strip_indices:
                continue  # e.g. no z_strip wave for a 2D run

            self._create_class_ids(strip_indices)
            neighbours_by_idx = {
                idx: self.reconciler.gather_neighbours(idx, finalized)
                for idx in strip_indices
            }
            finalized[chunk_type] = self._run_wave(
                engine, image, strip_indices, chunk_type, neighbours_by_idx,
                context, **kwargs
            )

        self.reconciler.apply_final_remap(finalized, context)
        result = self.reconciler.assemble_result(finalized, context)
        result = self.strategy.finalize(result, engine=engine, **kwargs)

        if not is_array_based:
            # Single rasterization pass -- the only point a dense array
            # gets written for tracker-based strategies (report section
            # 1/6). Array-based strategies never need this: chunks
            # already wrote directly into context['zarr_store'].
            from empanada.inference.patterns import fill_panoptic_volume

            trackers_to_fill = result.trackers_dict.get('consensus')
            if trackers_to_fill is None:
                trackers_to_fill = [
                    tr for trackers in result.trackers_dict.values() for tr in trackers
                ]

            zout = _write_empty_chunk(self.zarr_outpath, image, inp_scale=[0.005, 0.005])
            fill_panoptic_volume(zout, trackers_to_fill)
            context['zarr_store'] = zout

        if self.multiscale_writer is not None:
            self.multiscale_writer(self.zarr_inpath, image, self.zarr_outpath, image)
            return da.from_zarr(self.zarr_outpath + "/labels/seg")

        # No multiscale pyramid writer supplied: hand back the reconciled
        # zarr array directly rather than a multiscale pyramid.
        return da.from_zarr(context['zarr_store'])
