import os
import time
import zarr
import torch
import numpy as np
import dask.array as da

from empanada.seg_executors import SingleRegionExecutor, ChunkedExecutor, StackStrategy, OrthoplaneStrategy
from empanada_napari.inference import Engine3d
from empanada_napari.multigpu import MultiGPUEngine3d
from empanada_napari.utils import get_configs, abspath
from empanada.config_loaders import read_yaml


quantized_supported = True
if torch.backends.quantized.engine in (None or 'none'):
    quantized_supported = False

class VolumeSegPipeline:
    def __init__(self,
            image: np.ndarray | da.Array | zarr.Array,
            model_config: str,
            multiscale_level: int=0,
            use_gpu: bool = False,
            use_quantized: bool = False,
            multigpu: bool = False,

            downsampling: int = 1,
            confidence_thr: float = 0.5,
            center_confidence_thr: float = 0.1,
            min_distance_object_centers: int = 3,
            fine_boundaries: bool = False,
            semantic_only: bool = False,

            median_slices: int = 3,
            min_size: int = 500,
            min_extent: int = 5,
            maximum_objects_per_class: str = '10000',
            inference_plane: str = 'xy',

            label_erosion: int = 0,
            label_dilation: int = 0,
            fill_holes_in_segmentation: bool = False,
            orthoplane: bool = False,
            return_panoptic: bool = False,
            pixel_vote_thr: int = 2,
            allow_one_view: bool = False,

            use_store_dir: bool = False,
            store_dir: str = None,
            chunk_size: int|list[int] = 256
    ):
        
        self.image = image
        self.multiscale_level = multiscale_level
        self.model_config_name = model_config
        self.use_gpu = use_gpu
        self.use_quantized = use_quantized
        self.multigpu = multigpu

        self.downsampling = downsampling
        self.confidence_thr = confidence_thr
        self.center_confidence_thr = center_confidence_thr
        self.min_distance_object_centers = min_distance_object_centers
        self.fine_boundaries = fine_boundaries
        self.semantic_only = semantic_only

        self.median_slices = median_slices
        self.min_size = min_size
        self.min_extent = min_extent
        self.maximum_objects_per_class = int(maximum_objects_per_class)
        self.inference_plane = inference_plane

        self.label_erosion = label_erosion
        self.label_dilation = label_dilation
        self.fill_holes = fill_holes_in_segmentation
        self.orthoplane = orthoplane
        self.return_panoptic = return_panoptic
        self.pixel_vote_thr = pixel_vote_thr
        self.allow_one_view = allow_one_view

        self.use_store_dir = use_store_dir
        self.store_dir = str(store_dir)
        self.last_config = None
        self.engine = None

        if isinstance(chunk_size, str):
            chunk_size = [int(s) for s in chunk_size.split(',')]

        if type(chunk_size) == int:
            self.chunk_size = tuple(chunk_size for _ in range(3))
        elif len(chunk_size) == 1:
            self.chunk_size = tuple(int(chunk_size[0]) for _ in range(3))
        else:
            assert len(chunk_size) == 3, f"Chunk size must be 1 or 3 integers, got {chunk_size}"
            self.chunk_size = tuple(int(s) for s in chunk_size)

        self.image_name = 'labels'

        self._check_option_compatibility()

        self.image = self._preprocess_image_array()

    # ---------------- Pipeline running entrypoint ----------------
    def run(self, zarr_inpath=None, zarr_outpath=None):  
        '''Step 1: Load the model config'''
        model_configs = get_configs()
        self.model_config = read_yaml(model_configs[self.model_config_name])

        if self.last_config is None:
            self.last_config = self.model_config_name

        '''Step 2: Create storage url from layer name and model config'''
        if not self.use_store_dir: # This is a default -
            self.store_url = None
            print(f'Running without zarr storage directory, this may use a lot of memory!')
        else:
            # Consider using this url for both: regular Zarr output & OME-Zarr output
            self.store_url = os.path.join(self.store_dir, f'{self.image_name}_{self.model_config_name}.zarr')

        '''Step 3: Setup the Engine'''
        self.get_engine()

        '''Step 4: Get the 3d slice from the image array'''
        # image = self._preprocess_image_array()
        image = self.image
        print(image.shape, self.image.shape)

        '''Step 5: Pick the InferenceStrategy (batch vs single-slice) and
        the Executor (chunked vs single-region) for this run.'''
        strategy = self._select_strategy()
        executor = self._select_executor(strategy, image, zarr_inpath, zarr_outpath)

        '''Step 6: Return the computed segmentation array'''
        seg, axis_name = executor.run_workflow(self.engine, image, self.inference_plane)

        return seg, axis_name

    # ---------------- Engine management ----------------
    def get_engine(self):
        reload_engine = (
            self.engine is None
            or self.last_config != self.model_config_name
        )

        if reload_engine:
            self.engine = Engine3d(
                self.model_config,
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                use_gpu=self.use_gpu,
                use_quantized=self.use_quantized,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size,
                label_erosion=self.label_erosion,
                label_dilation=self.label_dilation,
                fill_holes_in_segmentation=self.fill_holes
            )
            self.last_config = self.model_config_name
            self.using_gpu = self.use_gpu

        elif self.multigpu:
            self.engine = MultiGPUEngine3d(
                self.model_config,
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size
            )
            self.last_config = self.model_config_name

        else:
            # update the parameters
            self.engine.update_params(
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size,
                label_erosion=self.label_erosion,
                label_dilation=self.label_dilation,
                fill_holes_in_segmentation=self.fill_holes
            )
        return

    # ---------------- Helper methods ----------------
    
    def _select_strategy(self):
        if self.orthoplane:
            return OrthoplaneStrategy()
        return StackStrategy()

    def _select_executor(self, strategy, image, zarr_inpath, zarr_outpath):
        # zarr_inpath/zarr_outpath are optional, not gating conditions:
        # ChunkedExecutor's _write_empty_chunk passes zarr_outpath straight
        # to zarr.open_group, which falls back to an in-memory store when
        # it's None -- so a lazy (dask/zarr) image gets chunked whether or
        # not the caller supplied explicit store paths.
        if isinstance(image, (da.Array, zarr.Array)):
            scale = 2
            return ChunkedExecutor(strategy, zarr_inpath, zarr_outpath, scale=scale)
        return SingleRegionExecutor(strategy)

    def _check_option_compatibility(self):
        if quantized_supported == False and self.use_quantized:
            raise RuntimeWarning(
                f" No quantized backend is selected. torch.backends.quantized.engine = {torch.backends.quantized.engine} Using Quantized Model may fail."
            )
        return
    
    def _preprocess_image_array(self):
        image = self.image
        # Get the 3d slice from the image 

        # Verify that the image doesn't have extraneous channel dimensions
        assert image.ndim in [3, 4], "Only 3D and 4D input images can be handled!"
        if image.ndim == 4:
            # Channel dimensions are commonly 1, 3 and 4
            # Check for dimensions on zeroth and last axes
            shape = image.shape
            if shape[0] in [1, 3, 4]:
                image = image[0]
            elif shape[-1] in [1, 3, 4]:
                image = image[..., 0]
            else:
                raise Exception(f'Image volume must be 3D, got image of shape {shape}')

            print(f'Got 4D image of shape {shape}, extracted single channel of size {image.shape}')

        return image

