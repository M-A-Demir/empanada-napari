import os
import time
import zarr
import torch
import napari
import numpy as np
import dask.array as da

from napari import Viewer
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from magicgui import widgets, magic_factory
from qtpy.QtWidgets import QScrollArea

from pathlib import Path
from itertools import product
from torch.cuda import device_count
from empanada.inference.volume_inference import VolumeSegPipeline

quantized_supported = True
if torch.backends.quantized.engine in (None or 'none'):
    quantized_supported = False


class VolumeSegPipelineGUI(VolumeSegPipeline):
    def __init__(self, image_layer: Image, viewer: Viewer, pbar: widgets.ProgressBar = None,
                 *args, **kwargs):
        image = self._get_image_layer_as_array(image_layer)
        self.image_layer = image_layer
        self.image_name = image_layer.name
        self.viewer = viewer
        self.pbar = pbar

        super().__init__(image, *args, **kwargs)

    # ---------------- (Threaded) Pipeline running entrypoint ----------------
    @thread_worker
    def run_in_thread(self, zarr_inpath=None, zarr_outpath=None):
        return self.run(zarr_inpath, zarr_outpath)

    # ---------------- Helper methods ----------------    
    def _get_image_layer_as_array(self, image_layer):
        return image_layer.data

    def _preprocess_image_array(self):
        if self.image_layer.multiscale:
            print(f'Multiscale image selected, using resolution level {self.multiscale_level}!')
            try:
                if self.multiscale_level <= len(self.image):
                    self.image = self.image[self.multiscale_level]
            except IndexError:
                raise Exception(f'Maximum multiscale level is {len(self.image) - 1}, got multiscale level {self.multiscale_level}')
      
        return super()._preprocess_image_array()

    # ---------------- GUI result output functions ----------------
    def _new_layers(self, mask, description, instances=None):
        metadata = {}
        if instances is not None:
            for label, label_attrs in instances.items():
                metadata[label] = {
                    'box': label_attrs['box'],
                    'area': label_attrs['runs'].sum(),
                }
        translate = self.image_layer.translate
        scale = self.image_layer.scale
        ndim = self.image_layer.data[0].ndim if self.image_layer.multiscale else self.image_layer.data.ndim
        if ndim:
            shape = self.image_layer.data.shape
            if shape[0] in [1, 3, 4]:
                translate = translate[1:]
                scale = scale[1:]
            elif shape[-1] in [1, 3, 4]:
                translate = translate[:-1]
                scale = scale[:-1]

            if self.multiscale_level > 0:
                if shape[0] in [1,3,4]:
                    shape_full = np.asarray(self.image_layer.data.shapes[0][1:])
                    shape_scaled = np.asarray(self.image_layer.data.shapes[self.multiscale_level][1:])
                if shape[-1] in [1,3,4]:
                    shape_full = np.asarray(self.image_layer.data.shapes[0][:-1])
                    shape_scaled = np.asarray(self.image_layer.data.shapes[self.multiscale_level][:-1])

                scale = scale * shape_full / shape_scaled

        self.viewer.add_labels(
            mask, name=f'{self.image_layer.name}-{description}',
            visible=True, metadata=metadata, translate=translate,
            scale=scale
        )
        self.pbar.hide()

    def new_segmentation(self, *args):
        mask = args[0][0]
        axis_name = args[0][1]
        if mask is not None:
            try:
                self._new_layers(mask, f'panoptic-stack-{axis_name}')
                for layer in self.viewer.layers:
                    layer.visible = False
                self.viewer.layers[-1].visible = True
                self.image_layer.visible = True
            except Exception as e:
                print(e)

    def new_class_stack(self, *args):
        masks, class_name, instances = args[0]
        try:
            self._new_layers(masks, f'{class_name}-prediction', instances)
            for layer in self.viewer.layers:
                layer.visible = False
            self.viewer.layers[-1].visible = True
            self.image_layer.visible = True
        except Exception as e:
            print(e)


def volume_inference_widget():
    # Import when users activate plugin
    from torch.cuda import device_count
    from empanada_napari.utils import get_configs, abspath

    logo = abspath(__file__, 'resources/empanada_logo.png')
    model_configs = get_configs()

    def on_init(widget):

        def set_max_multiscale(image_layer: Image):
            if image_layer.multiscale:
                widget.multiscale_level.visible = True
                widget.multiscale_level.max = len(image_layer.data) - 1
            else:
                widget.multiscale_level.visible = False

        def toggle_zarr_store_visibility(use_store_dir: bool):
            zarr_store_props = ['store_dir', 'chunk_size']
            if use_store_dir:
                for x in zarr_store_props:
                    setattr(getattr(widget, x), 'visible', True)
            else:
                for x in zarr_store_props:
                    setattr(getattr(widget, x), 'visible', False)
                widget.store_dir.set_value('')

        # Default settings
        if isinstance(widget.image_layer.value, Image):
            set_max_multiscale(image_layer=widget.image_layer.value)
        else:
            widget.multiscale_level.visible = False

        toggle_zarr_store_visibility(use_store_dir=False)

        widget.image_layer.changed.connect(set_max_multiscale)
        widget.use_store_dir.changed.connect(toggle_zarr_store_visibility)

    @magic_factory(
        widget_init=on_init,
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 3D Inference',
        layout='vertical',
        scrollable=True,

        multiscale_level=dict(widget_type='SpinBox', value=0, min=0, label='Multiscale level',
                             tooltip='What multiscale level should be segmented.'),

        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()),
                          value=list(model_configs.keys())[0], tooltip='Model to use for inference'),

        use_gpu=dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1,
                     tooltip='If checked, run on GPU 0'),
        use_quantized=dict(widget_type='CheckBox', text='Use quantized model', value=device_count()==0 and quantized_supported,
                           tooltip='If checked, use the quantized model for faster CPU inference.'),
        multigpu=dict(widget_type='CheckBox', text='Multi GPU', value=False,
                      tooltip='If checked, run on all available GPUs'),

        parameters2d_head=dict(widget_type='Label', label=f'<h3 text-align="center">2D Parameters</h3>'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling',
                          tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1,
                            label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05,
                                   label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=35, step=1,
                                         label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine Boundaries', value=False,
                             tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic Only', value=False,
                           tooltip='Only run semantic segmentation for all classes.'),

        parameters_stack_head=dict(widget_type='Label', label=f'<h3 text-align="center">Stack Parameters</h3>'),
        median_slices=dict(widget_type='ComboBox', choices=[1, 3, 5, 7, 9, 11], value=3, label='Median Filter Size',
                           tooltip='Median filter size'),
        min_size=dict(widget_type='SpinBox', value=500, min=0, max=1e6, step=100, label='Min Size (Voxels)'),
        min_extent=dict(widget_type='SpinBox', value=5, min=0, max=1000, step=1, label='Min Box Extent'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='10000', label='Max objects per class in 3D',
                                       tooltip='Maximum number of objects per class in 3D inference'),
                                       # value here was originally '10000' string, may break
        inference_plane=dict(widget_type='ComboBox', choices=['xy', 'xz', 'yz'], value='xy', label='Inference plane',
                             tooltip='Image plane along which to run inference. Overwritten, if using ortho-plane.'),
        parameters_ortho_head=dict(widget_type='Label',
                                   label=f'<h3 text-align="center">Ortho-plane Parameters (Optional)</h3>'),
        label_erosion=dict(widget_type='SpinBox', value=0, min=0, max=50, step=1, label='Erode Labels',
                           tooltip='How much to erode labels produced after inference'),
        label_dilation=dict(widget_type='SpinBox', value=0, min=0, max=50, step=1, label='Dilate Labels',
                            tooltip='How much to dilate labels produced after inference'),
        fill_holes_in_segmentation=dict(widget_type='CheckBox', text='Fill holes in segmentation', value=False,
                                            tooltip='Whether to fill holes in the segmentation after inference'),
        orthoplane=dict(widget_type='CheckBox', text='Run ortho-plane', value=False,
                        tooltip='Whether to run orthoplane inference'),
        return_panoptic=dict(widget_type='CheckBox', text='Return xy, xz, yz stacks', value=False,
                             tooltip='Whether to return the inference stacks.'),
        pixel_vote_thr=dict(widget_type='SpinBox', value=2, min=1, max=3, step=1, label='Voxel Vote Thr Out of 3',
                            tooltip='Number of votes out of 3 for a voxel to be labeled in the consensus'),
        allow_one_view=dict(widget_type='CheckBox', text='Permit detections found in 1 stack into consensus',
                            value=False,
                            tooltip='Whether to allow detections into consensus that were picked up by inference in just 1 stack'),

        storage_head=dict(widget_type='Label', label=f'<h3 text-align="center">Zarr Storage (optional)</h3>'),
        use_store_dir=dict(widget_type='CheckBox', text='Use Zarr Storage', value=False,
                           tooltip='Whether to select local Zarr storage'),
        store_dir=dict(widget_type='FileEdit', value='', label='Directory', mode='d',
                       tooltip='location to store segmentations on disk'),
        chunk_size=dict(widget_type='LineEdit', value='256', label='Chunk size',
                        tooltip='Chunk size of the zarr array. Integer or comma separated list of 3 integers.'),
        pbar={'visible': False, 'max': 0, 'label': 'Running...'},
    )
    def widget_factory(
            viewer: napari.viewer.Viewer,
            label_head,
            image_layer: Image,
            multiscale_level,
            model_config,
            use_gpu,
            use_quantized,
            multigpu,

            parameters2d_head,
            downsampling,
            confidence_thr,
            center_confidence_thr,
            min_distance_object_centers,
            fine_boundaries,
            semantic_only,

            parameters_stack_head,
            median_slices,
            min_size,
            min_extent,
            maximum_objects_per_class,
            inference_plane,

            parameters_ortho_head,
            label_erosion,
            label_dilation,
            fill_holes_in_segmentation,
            orthoplane,
            return_panoptic,
            pixel_vote_thr,
            allow_one_view,

            storage_head,
            use_store_dir,
            store_dir,
            chunk_size,

            pbar: widgets.ProgressBar
    ):
        # instantiate the class
        pipeline = VolumeSegPipelineGUI(viewer = viewer,
            image_layer = image_layer,
            model_config = model_config,
            use_gpu = use_gpu,
            use_quantized = use_quantized,
            multigpu = multigpu,
            downsampling = downsampling,
            confidence_thr = confidence_thr,
            center_confidence_thr = center_confidence_thr,
            min_distance_object_centers = min_distance_object_centers,
            fine_boundaries = fine_boundaries,
            semantic_only = semantic_only,
            median_slices = median_slices,
            min_size = min_size,
            min_extent = min_extent,
            maximum_objects_per_class = maximum_objects_per_class,
            inference_plane = inference_plane,
            multiscale_level=multiscale_level,
            label_erosion = label_erosion,
            label_dilation = label_dilation,
            fill_holes_in_segmentation = fill_holes_in_segmentation,
            orthoplane = orthoplane,
            return_panoptic = return_panoptic,
            pixel_vote_thr = pixel_vote_thr,
            allow_one_view = allow_one_view,
            use_store_dir=use_store_dir,
            store_dir = store_dir,
            chunk_size = chunk_size,
            pbar = pbar)

        # method that configures & runs inference
        worker = pipeline.run_in_thread()

        if orthoplane:
            worker.returned.connect(pipeline.new_segmentation)
            worker.returned.connect(pipeline.new_class_stack)      

        else:
            worker.returned.connect(pipeline.new_segmentation)
            worker.returned.connect(pipeline.new_class_stack)

        pbar.show()

    # instantiate widget
    widget = widget_factory()

    # make the scroll available
    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def volume_dock_widget():
    return volume_inference_widget, {'name': '3D Inference'}
