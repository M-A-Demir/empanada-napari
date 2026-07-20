import numpy as np
from time import time
from tqdm import tqdm
from skimage.draw import polygon

from empanada.inference.slice_inference import SliceSegPipeline
from empanada_napari.utils import get_configs, abspath

from napari import Viewer
from napari.layers import Image, Labels, Shapes
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QScrollArea
from magicgui import magicgui, widgets

from torch.cuda import device_count
from torch.backends.quantized import engine


quantized_supported = True
if engine in (None or 'none'):
    quantized_supported = False


class SliceSegPipelineGUI(SliceSegPipeline):
    def __init__(self, image_layer: Image, viewer: Viewer, viewport: bool = False,
                 output_to_layer: bool = False, output_layer: Labels = None, pbar: widgets.ProgressBar = None,
                 *args, **kwargs):
        image = self._get_image_layer_as_array(image_layer)
        self.image_layer = image_layer
        self.viewer = viewer
        self.viewport = viewport
        self.output_to_layer = output_to_layer
        self.output_layer = output_layer
        self.pbar = pbar

        super().__init__(image, *args, **kwargs)

    # ---------------- (Threaded) Pipeline running entrypoint ----------------
    @thread_worker
    def run_in_thread(self, zarr_inpath=None, zarr_outpath=None):
        return self.run(zarr_inpath, zarr_outpath)

    # ---------------- Helper methods ----------------    
    def _get_image_layer_as_array(self, image_layer):
        return image_layer.data

    def _check_option_compatibility(self):
        super()._check_option_compatibility()

        if self.output_to_layer:
            assert self.output_layer is not None, "Must select an output layer or uncheck Output to layer!"
            assert self.output_layer.data.shape == self.image_layer.data.shape, \
                "Output layer must have the same shape as the input image."
            assert self.viewport is False, "Cannot output to layer and restrict to viewport at the same time."

        if self.batch_mode:
            assert self.viewport is False, "Cannot use batch mode and restrict to viewport at the same time."
            assert not self.output_to_layer, "Batch mode is not compatible with output to layer!"
            assert not self.image_layer.multiscale, "Batch mode is not compatible with multiscale images!"
            assert not self.viewport, "Batch mode is not compatible with viewport inference!"
            assert not self.confine_to_roi, "Batch mode is not compatible with ROI inference!"

        if self.viewport:
            assert all(s == 1 for s in
                       self.image_layer.scale), "Viewport inference only supports images with scale 1 in all dimensions!"
            assert self.viewer.dims.order[0] != 1, "Viewport inference not supported for xz planes!"

        if not all(s == 1 for s in self.image_layer.scale):
            print(f'Image has non-unit scale. 2D segmentations will disappear after rotation or axis rolling!')
        
        return

    def _preprocess_image_array(self):
        # Get the 2d slice from the image
        if self.batch_mode:
            return super()._preprocess_image_array()
        
        if self.confine_to_roi:
            shapes_layer = [layer for layer in self.viewer.layers if isinstance(layer, Shapes)][0]
            image, y, x, y_max, x_max, binary_mask = self._get_roi_slice(self.image, shapes_layer)
            image[binary_mask == False] = 0
            axis, plane = "overloaded", self.image_layer.data.shape
        else:
            image, axis, plane, y, x = self._get_current_slice(self.image, self.image_layer)

        print(f'Image of size {image.shape} sliced at plane {plane} from axis {axis}')
        return image, axis, plane, y, x        

    def _get_roi_slice(self, image, shapes_layer):
        shapes = np.array(shapes_layer.data)
        min_y, min_x = np.inf, np.inf
        max_y, max_x = -np.inf, -np.inf
        for shape in shapes:
            min_y, min_x = min(min_y, shape[:, 0].min()), min(min_x, shape[:, 1].min())
            max_y, max_x = max(max_y, shape[:, 0].max()), max(max_x, shape[:, 1].max())
        min_y, min_x, max_y, max_x = map(int, (min_y, min_x, max_y, max_x))
        roi = image[min_y:max_y, min_x:max_x].copy()
        mask = self._get_mask_from_roi(image, shapes_layer)
        return roi, min_y, min_x, max_y, max_x, mask[min_y:max_y, min_x:max_x]
    
    def _get_mask_from_roi(self, image, shapes_layer):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for shape in shapes_layer.data:
            rr, cc = polygon(shape[:, 0], shape[:, 1], (h, w))
            mask[rr, cc] = True
        return mask
    
    def _get_current_slice(self, image, image_layer):
        cursor_pos = self.viewer.cursor.position

        # handle multiscale by taking highest resolution level
        if image_layer.multiscale:
            print('Using highest resolution level from multiscale!')
            image = image[0]

        y, x = 0, 0
        if image.ndim == 4:
            axis = tuple(self.viewer.dims.order[:2])
            plane = (
                int(image_layer.world_to_data(cursor_pos)[axis[0]]),
                int(image_layer.world_to_data(cursor_pos)[axis[1]])
            )

            slices = [slice(None), slice(None), slice(None), slice(None)]

            slices[axis[0]] = plane[0]
            slices[axis[1]] = plane[1]
            if self.viewport:
                yslice, xslice = self._viewer_slices(image_layer, plane)
                slices[2] = yslice
                slices[3] = xslice
                y = yslice.start
                x = xslice.start

        elif image.ndim == 3:
            axis = self.viewer.dims.order[0]
            plane = int(image_layer.world_to_data(cursor_pos)[axis])

            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = plane
            if self.viewport:
                yaxis, xaxis = [i for i in range(3) if i != axis]
                yslice, xslice = self._viewer_slices(image_layer, plane, axis)
                slices[yaxis] = yslice
                slices[xaxis] = xslice
                y = yslice.start
                x = xslice.start
                print(f'Slices {slices}')

        else:
            slices = [slice(None), slice(None)]
            axis = None
            plane = None
            if self.viewport:
                yslice, xslice = self._viewer_slices(image_layer, plane, axis)
                slices[0] = yslice
                slices[1] = xslice
                y = yslice.start
                x = xslice.start

        return image[tuple(slices)], axis, plane, y, x
    
    def _viewer_slices(self, image_layer, plane=None, axis=None):
        corners = image_layer.corner_pixels.T.tolist()
        if isinstance(axis, tuple) and isinstance(plane, tuple):
            yslice = slice(*corners[2])
            xslice = slice(*corners[3])
        elif axis is not None:
            # handle all of the weird special cases
            cases12 = [(0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
            cases01 = [(1, 2, 0)]
            cases20 = [(2, 0, 1)]
            if self.viewer.dims.order in cases12:
                yslice = slice(*corners[1])
                xslice = slice(*corners[2])
            elif self.viewer.dims.order in cases01:
                yslice = slice(*corners[0])
                xslice = slice(*corners[1])
            elif self.viewer.dims.order in cases20:
                yslice = slice(*corners[2])
                xslice = slice(*corners[0])
        else:
            yslice = slice(*corners[0])
            xslice = slice(*corners[1])

        print(f'Corners {corners}, slices {yslice, xslice}')

        return yslice, xslice
    
    # ---------------- GUI result output functions ----------------
    def show_batch_stack(self, *args):
        stack = args[0]
        self.viewer.add_labels(stack, name=self.image_layer.name + '_batch_segs')
        self.pbar.hide()

    def show_result(self, *args):
        seg, axis, plane, y, x = args[0]

        if axis == "overloaded":
            out_2d = np.zeros(plane, dtype=seg.dtype)
            seg_shape = seg.shape
            out_2d[y:y + seg_shape[0], x:x + seg_shape[1]] = seg
            seg = out_2d
            translate = [0, 0]
        elif axis is not None and plane is not None:
            if isinstance(axis, tuple) and isinstance(plane, tuple):
                seg = np.expand_dims(seg, axis=axis)
                translate = [0, 0, y, x]
                translate[axis[0]] = plane[0]
                translate[axis[1]] = plane[1]
            else:
                seg = np.expand_dims(seg, axis=axis)

                # oddly translate has to be a list and
                # not an array or things break. WHY????
                translate = self.image_layer.translate.tolist()
                translate[axis] += plane
                yaxis, xaxis = [i for i in range(3) if i != axis]
                if y is not None:
                    translate[yaxis] += y
                if x is not None:
                    translate[xaxis] += x
        else:
            translate = [y, x]

        self.viewer.add_labels(seg, name=f'empanada_seg_2d', visible=True, translate=tuple(translate))
        self.viewer.layers[-1].scale = self.image_layer.scale

        self.pbar.hide()

    def store_result(self, *args):
        seg, axis, plane, y, x = args[0]

        if axis == "overloaded":
            out_2d = np.zeros(plane, dtype=seg.dtype)
            seg_shape = seg.shape
            out_2d[y:y + seg_shape[0], x:x + seg_shape[1]] = seg
            seg = out_2d
            self.output_layer.data = seg
        elif axis is not None and plane is not None:
            if isinstance(axis, tuple) and isinstance(plane, tuple):
                # 4D flipbook case
                slices = [slice(None), slice(None), slice(None), slice(None)]
                slices[axis[0]] = plane[0]
                slices[axis[1]] = plane[1]
                self.output_layer.data[tuple(slices)] = seg
            else:
                # 3D case
                slices = [slice(None), slice(None), slice(None)]
                slices[axis] = plane
                self.output_layer.data[tuple(slices)] = seg
        else:
            # 2D case
            self.output_layer.data = seg

        # self.output_layer.visible = False
        self.output_layer.visible = True

        self.pbar.hide()


# ---------------- Napari GUI wrapper ----------------
def slice_inference_widget():
    """
    Factory function to create the widget for Napari.
    This is what Napari will call.
    """

    logo = abspath(__file__, 'resources/empanada_logo.png')
    model_configs = get_configs()

    # define magicgui params
    gui_params = dict(
        model_config=dict(widget_type='ComboBox', choices=list(model_configs.keys()),
                          value=list(model_configs.keys())[0], label='Model', tooltip='Model to use for inference'),
        # store_dir=dict(widget_type='FileEdit', value='no zarr storage', label='Directory', mode='d',
                    #    tooltip='location to store segmentations on disk'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling',
                          tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1,
                            label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05,
                                   label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=35, step=1,
                                         label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine boundaries', value=False,
                             tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic only', value=False,
                           tooltip='Only run semantic segmentation for all classes.'),
        fill_holes_in_segmentation=dict(widget_type='CheckBox', text='Fill holes in segmentation', value=False,
                                        tooltip='If checked, fill holes in the segmentation mask.'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='10000', label='Max objects per class',
                                       tooltip='Maximum number of objects per class/ label divisor for mutliclass segmentation.'),
        tile_size=dict(widget_type='SpinBox', value=0, min=0, max=128000, step=1280, label='Tile size',
                       tooltip='Tile size for inference, whole image will be segmented if 0'),
        batch_mode=dict(widget_type='CheckBox', text='Batch mode', value=False,
                        tooltip='If checked, each image in a stack is segmented independently.'),
        viewport=dict(widget_type='CheckBox', text='Confine to viewport', value=False,
                      tooltip='If checked, inference will be restricted to the current viewport.'),
        output_to_layer=dict(widget_type='CheckBox', text='Output to layer', value=False,
                             tooltip='If checked, the segmentation is output to the selected output layer.'),
        use_gpu=dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1,
                                 tooltip='If checked, run on GPU 0'),
        use_quantized=dict(widget_type='CheckBox', text='Use quantized model', value=device_count() == 0 and quantized_supported,
                                       tooltip='If checked, run on GPU 0'),
        confine_to_roi=dict(widget_type='CheckBox', text='Confine to ROI', value=False,
                                        tooltip='If checked, inference will be restricted to the ROI defined by a shapes layer.')
    )

    
    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 2D Inference',
        layout='vertical',
        scrollable=True,
        pbar={'visible': False, 'max': 0, 'label': 'Running...'},
        **gui_params
    )
    def widget(
            viewer: Viewer,
            label_head,
            image_layer: Image,
            model_config,
            downsampling,
            confidence_thr,
            center_confidence_thr,
            min_distance_object_centers,
            fine_boundaries,
            semantic_only,
            fill_holes_in_segmentation,
            maximum_objects_per_class,
            tile_size,
            batch_mode,
            use_gpu,
            use_quantized,
            viewport,
            confine_to_roi,
            output_to_layer,
            output_layer: Labels,
            pbar: widgets.ProgressBar
    ):

        # instantiate the class
        pipeline = SliceSegPipelineGUI(viewer=viewer,
            image_layer=image_layer,
            model_config=model_config,
            downsampling=downsampling,
            confidence_thr=confidence_thr,
            center_confidence_thr=center_confidence_thr,
            min_distance_object_centers=min_distance_object_centers,
            fine_boundaries=fine_boundaries,
            semantic_only=semantic_only,
            fill_holes_in_segmentation=fill_holes_in_segmentation,
            maximum_objects_per_class=maximum_objects_per_class,
            tile_size=tile_size,
            batch_mode=batch_mode,
            use_gpu=use_gpu,
            use_quantized=use_quantized,
            viewport=viewport,
            confine_to_roi=confine_to_roi,
            output_to_layer=output_to_layer,
            output_layer=output_layer,
            pbar=pbar)
        
        # method that configures & runs inference
        worker = pipeline.run_in_thread()

        if batch_mode:
            worker.returned.connect(pipeline.show_result if image_layer.data.ndim == 2 \
                                    else pipeline.show_batch_stack)
        else:
            worker.returned.connect(pipeline.store_result if output_to_layer \
                                    else pipeline.show_result)
        worker.start()
        pbar.show()

    # make the scroll available
    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll
    
    return widget


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def slice_dock_widget():
    return slice_inference_widget, {'name': '2D Inference (Parameter Testing)'}
