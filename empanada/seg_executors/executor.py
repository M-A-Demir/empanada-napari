from abc import ABC, abstractmethod
from napari.qt.threading import thread_worker
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import binary_fill_holes

import dask.array as da
import numpy as np

class Executor(ABC):
    def __init__(self,
                 fill_holes_in_segmentation=False,
                 orthoplane=False
                 ):
        self.fill_holes = fill_holes_in_segmentation
        self.orthoplane = orthoplane

    @abstractmethod
    def run_workflow(self):
        return NotImplementedError
    
    # ---------------- Inference runners ----------------
    def _get_segmentation(self, engine, image=None, axis=None, plane=None, y=None, x=None):
        # To Do: Decide what to do here, do we process 'result' into also returning seg, axis, etc.?

        if image.ndim == 2:
            seg, axis, plane, y, x = self._get_segmentation_2d(engine, image, axis, plane, y, x)
            return seg, axis, plane, y, x

        elif image.ndim == 3 or image.ndim == 4: #?
            result = self._get_segmentation_3d(engine, image, plane)
            return result

    def _get_segmentation_2d(self, engine, image=None, plane=None):
        
        if self.batch_mode:
            seg, axis, plane, y, x = self._run_model_batch(engine, image, self.fill_holes)
        else:
            seg, axis, plane, y, x = self._run_model(engine, image, axis, plane, y, x, self.fill_holes)

        return seg, axis, plane, y, x
    

    def _get_segmentation_3d(self, engine, image=None, plane=None):
        # This part needs super refactoring
        if self.orthoplane:
            result = self._orthoplane_inference(engine, image)
        else:
            result = self._stack_inference(engine, image, plane)

        return result

    # ---------------- 2D ----------------
    @thread_worker
    def run_model(self, engine, image, axis, plane, y, x, fill_holes):
        return self._run_model(engine, image, axis, plane, y, x, fill_holes)

    @thread_worker
    def run_model_batch(self, engine, image, fill_holes):
        return self._run_model_batch(engine, image, fill_holes)
    
    def _run_model(self, engine, image, axis, plane, y, x, fill_holes):
        # create the inference engine
        seg = engine.infer(image)
        if fill_holes:
            seg = self._fill_holes_in_segmentation(seg)
        return seg, axis, plane, y, x

    def _run_model_batch(self, engine, image, fill_holes):
        # axis is always xy
        axis = 0

        # create the inference engine
        if image.ndim == 3:
            print(f'Running batch mode inference on {len(image)} images.')
            segmentations = []
            for plane, img_slice in tqdm(enumerate(image), total=len(image)):
                if type(img_slice) == da.core.Array:
                    img_slice = img_slice.compute()

                seg = engine.infer(img_slice)
                if fill_holes:
                    seg = self._fill_holes_in_segmentation(seg)
                segmentations.append(seg)

            # stack segmentations with padding
            max_h = max(seg.shape[0] for seg in segmentations)
            max_w = max(seg.shape[1] for seg in segmentations)
            padded = []
            for seg in segmentations:
                h, w = seg.shape
                padh, padw = max_h - h, max_w - w
                padded.append(np.pad(seg, ((0, padh), (0, padw))))

            padded = np.stack(padded, axis=0)
            return padded

        elif image.ndim == 2:
            if type(image) == da.core.Array:
                image = image.compute()

            plane = 0
            seg = engine.infer(image)
            if fill_holes:
                seg = self._fill_holes_in_segmentation(seg)
            return seg, None, None, None, None
        
        else:
            raise Exception(f'Batch mode supports 2d and 3d, got {image.ndim}d.')

    # ---------------- 3D ----------------

    @thread_worker
    def stack_inference(self, engine, volume, axis_name):
        return self._stack_inference(engine, volume, axis_name)

    @thread_worker
    def orthoplane_inference(self, engine, volume):
        return self._orthoplane_inference(engine, volume)

    def _stack_inference(self, engine, volume, axis_name):
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        trackers_dict = {axis_name: trackers}
        return stack, axis_name, trackers_dict

    def _orthoplane_inference(self, engine, volume):
        trackers_dict = {}
        axes_dict = {}
        axes_dict = {}
        for axis_name in ['xy', 'xz', 'yz']:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            trackers_dict[axis_name] = trackers
            
            # report instances per class
            for tracker in trackers:
                class_id = tracker.class_id
                print(f'Class {class_id}, axis {axis_name}, has {len(tracker.instances.keys())} instances')
            axes_dict[axis_name] = stack
        return trackers_dict, axes_dict

    # ---------------- Helper methods ----------------
    def _fill_holes_in_segmentation(self, mask):
        # This lives on the shared Executor base class, but only the 2D
        # (dense-array) codepaths above ever call it. skimage.measure.regionprops
        # requires a dense labeled array, which tracker-based 3D subclasses
        # (stack/orthoplane inference) never produce here -- calling this on
        # tracker output would fail confusingly rather than at this clear
        # boundary. Asserting ndim==2 turns a future accidental 3D call into
        # a loud, immediate error instead of a silent one deep inside
        # regionprops. The real fix is architectural (move this method off
        # the shared base class entirely, onto the array-only strategy) --
        # see report-2026-07-16-segmentation-architecture.md.
        assert mask.ndim == 2, (
            f'_fill_holes_in_segmentation only supports 2D dense label arrays, got {mask.ndim}D. '
            'This method is only valid for array-based (2D) segmentation results.'
        )
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
    
