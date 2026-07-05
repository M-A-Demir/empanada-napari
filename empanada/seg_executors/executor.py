from abc import ABC, abstractmethod
from napari.qt.threading import thread_worker
from tqdm import tqdm
from skimage import measure
from scipy.ndimage import binary_fill_holes

import dask.array as da
import numpy as np

class Executor(ABC):
    def __init__(self,
                 fill_holes_in_segmentation=False
                 ):
        self.fill_holes = fill_holes_in_segmentation

    @abstractmethod
    def run_workflow(self):
        return NotImplementedError
    
    # ---------------- Inference runners ----------------
    def _get_segmentation(self, engine, image=None, axis=None, plane=None, y=None, x=None):
        
        if self.batch_mode:
            seg, axis, plane, y, x = self._run_model_batch(engine, image, self.fill_holes)
        else:
            seg, axis, plane, y, x = self._run_model(engine, image, axis, plane, y, x, self.fill_holes)

        return seg, axis, plane, y, x

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


    # ---------------- Helper methods ----------------    
    def _fill_holes_in_segmentation(self, mask):
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
    
