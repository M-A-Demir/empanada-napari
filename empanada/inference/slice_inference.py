import numpy as np
import dask.array as da
import zarr
from time import time

from empanada.seg_executors import SingleRegionExecutor, ChunkedExecutor, SingleSliceStrategy, BatchSliceStrategy
from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine2d
from empanada_napari.utils import get_configs, abspath

from torch.backends.quantized import engine

quantized_supported = True
if engine in (None or 'none'):
    quantized_supported = False


class SliceSegPipeline:
    def __init__(self, 
            image: np.ndarray | da.Array | zarr.array,
            model_config: str,
            downsampling: int = 1,
            confidence_thr: float = 0.5,
            center_confidence_thr: float = 0.1,
            min_distance_object_centers: int = 3,
            fine_boundaries: bool = False,
            semantic_only: bool = False,
            fill_holes_in_segmentation: bool = False,
            maximum_objects_per_class: int = 10000,
            tile_size: int = 0,
            batch_mode: bool = False,
            use_gpu: bool = False,
            use_quantized: bool = False,
            confine_to_roi: bool = False,
    ):
        self.image = image
        self.model_config_name = model_config
        self.downsampling = downsampling
        self.confidence_thr = confidence_thr
        self.center_confidence_thr = center_confidence_thr
        self.min_distance_object_centers = min_distance_object_centers
        self.fine_boundaries = fine_boundaries
        self.fill_holes = fill_holes_in_segmentation
        self.tile_size = tile_size
        self.batch_mode = batch_mode
        self.semantic_only = semantic_only
        self.using_gpu = use_gpu
        self.using_quantized = use_quantized
        self.confine_to_roi = confine_to_roi
        self.maximum_objects_per_class = int(maximum_objects_per_class)
        self.last_config = None
        self.engine = None

        self._check_option_compatibility()

    # ---------------- Pipeline running entrypoint ----------------
    def run(self, zarr_inpath=None, zarr_outpath=None):  
        '''Step 1: Load the model config'''
        model_configs = get_configs()
        self.model_config = read_yaml(model_configs[self.model_config_name])

        if self.last_config is None:
            self.last_config = self.model_config_name

        '''Step 2: Setup the Engine'''
        self.get_engine()
                
        '''Step 3: Get the 2D slice from the image array'''
        image, axis, plane, y, x = self._preprocess_image_array()
        print(image.shape, self.image.shape)

        '''Step 4: Pick the InferenceStrategy (batch vs single-slice) and
        the Executor (chunked vs single-region) for this run.'''
        strategy = self._select_strategy()
        executor = self._select_executor(strategy, image, zarr_inpath, zarr_outpath)

        '''Step 5: Return the computed segmentation array'''
        seg, axis, plane, y, x = executor.run_workflow(self.engine, image, axis=axis, 
                                                       plane=plane, y=y, x=x)

        return seg, axis, plane, y, x

    # ---------------- Engine Management ----------------
    def get_engine(self):
        reload_engine = (
            self.engine is None
            or self.last_config != self.model_config_name
        )

        if reload_engine:
            self.engine = Engine2d(
                self.model_config,
                inference_scale=self.downsampling,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                fine_boundaries=self.fine_boundaries,
                tile_size=self.tile_size,
                use_gpu=self.using_gpu,
                use_quantized=self.using_quantized,
            )
        else:
            # update the parameters of the engine
            # without reloading the model
            self.engine.update_params(
                inference_scale=self.downsampling,
                label_divisor=self.maximum_objects_per_class,
                nms_threshold=self.center_confidence_thr,
                nms_kernel=self.min_distance_object_centers,
                confidence_thr=self.confidence_thr,
                semantic_only=self.semantic_only,
                fine_boundaries=self.fine_boundaries,
                tile_size=self.tile_size,
            )
        self.last_config = self.model_config_name
        return

    # ---------------- Helper Methods ----------------
    def _select_strategy(self):
        if self.batch_mode:
            return BatchSliceStrategy(self.fill_holes)
        return SingleSliceStrategy(self.fill_holes)

    def _select_executor(self, strategy, image, zarr_inpath, zarr_outpath):
        if (type(image) == da.Array | zarr.array) and zarr_inpath and zarr_outpath:
            scale = 2
            return ChunkedExecutor(strategy, zarr_inpath, zarr_outpath, scale=scale)
        return SingleRegionExecutor(strategy)

    def _check_option_compatibility(self):
        if quantized_supported == False and self.using_quantized:
            raise RuntimeWarning(
                "No quantized backend is selected. " \
                f"torch.backends.quantized.engine = {engine}" \
                "Using Quantized Model may fail."
            )
        return
     
    def _preprocess_image_array(self):
        '''If input image is a 2D array, return
        '''
        image = self.image

    # Batch mode will iterate across a 3D array, so return array
    # Non-batch mode will run on 2D array, so return array if 2D, and slice if 3D
        if self.batch_mode:
            return image, None, None, None, None

        else:
            # if self.confine_to_roi:
                # Apply binary mask to the image
                # Not currently implemented outside of viewer
            
            # else:
            y, x = 0, 0
            slices = [slice(None)] * image.ndim
            axis = [0,1,2,3]

            if image.ndim == 4: # multiscale? use highest res level
                image = image[0]
                # axis = viewer param, as is plane so i don't think these are relevant. we will just slice along z
                axis = tuple(axis[:2])
                plane = (0, 0)
                slices[axis[0]], slices[axis[1]] = plane[0], plane[1]

            elif image.ndim == 3:
                axis = axis[0]
                # plane = 0
                plane = 223 # TMP
                slices[axis] = plane

            else:
                axis = None
                plane = None

            print(f'Image of size {image.shape} sliced at plane {plane} from axis {axis}')

            return image[tuple(slices)], axis, plane, y, x
