import numpy as np
import zarr
import math
import dask.array as da
import joblib

from .executor import Executor
from empanada.zarr_utils import _write_empty_chunk, _generate_tiles, _write_multiscale


class ParallelExecutor(Executor):
    def __init__(self,
                 zarr_inpath=None,
                 zarr_outpath=None,
                 fill_holes_in_segmentation=False
                 ):
        super().__init__(fill_holes_in_segmentation)
        self.zarr_inpath = zarr_inpath
        self.zarr_outpath = zarr_outpath
        self.class_ids = {}

    def _get_zarr_metadata(self):
        image_group = zarr.open_group(self.zarr_inpath, mode="r")

        # This should ... figure out what we need later


    def run_workflow(self, engine, image, scale=2, axis=None, plane=None, y=None, x=None):
        # First, get the downsampled array & chunks
        image = self._downsample_array(image, scale)
        tile_shape = [dim for dim in image.shape]
        chunk_indices = list(_generate_tiles(image.shape, tile_shape))

        # Get the class IDs:
        self._create_class_ids(chunk_indices)

        # Next, initialise the empty labels/tmp store/array
        # May want to pass whatever writer metadata we want to this func later
        zout_tmp = _write_empty_chunk(self.zarr_outpath, image, inp_scale=[0.005,0.005]) # inp_scale needs to come from input image zarr store

        # Call a method that can get seg, and update label (Change to delayed func later)
        self.process_segmentation_chunk(engine, image, axis, plane, y, x, 
                                   zout_tmp, slice_idx=None, 
                                   merge_labels=False)
        
        # Compute the strips across the panel seams
        strip_arrays = self._get_strips(chunk_sizes, padding=200)

        # Get their class IDs too:
        self._create_class_ids(strip_arrays)

        # Call the chunk processor in serial for strip arrays:
        for key in strip_arrays.keys():
            strip = strip_arrays[key]
            self.process_segmentation_chunk(engine, strip, axis, plane, y, x, 
                                   zout_tmp, slice_idx=None, 
                                   merge_labels=False)
            
        # Call ID reconciler
        tmp_array = self.zarr_outpath+"/labels/tmp/s0"
        self._reconcile_labels(tmp_array)    

        # Use this array to write the multiscales - Do I even need 'image' if I just pass the original image's shapes?
        self.write_out_multiscale(self.zarr_inpath, image, self.zarr_outpath, image)
        # maybe make write_out_multiscale return the seg path or array
        multiscale_arr_path = self.zarr_outpath+"/labels/seg"

        return da.from_zarr(multiscale_arr_path), axis, plane, y, x
    
    # --------- Joblib Delayed Methods ---------
    def delayed_process_segmentation_chunk(self, engine, image_down, chunk_indices, axis, plane, y, x, zout_down, merge_labels):
        compute = joblib.delayed(self.process_segmentation_chunk)
        jobs = [compute(image_down[idx], axis, plane, y, x, zout_down, idx) for idx in chunk_indices]
        
        for job in jobs: 
            print("Job:", job)

        executor = joblib.Parallel(n_jobs=-1, backend='threading')
        executor(jobs)
        return

    def delayed_apply_mapping(self, downseg, lut, zout_down, chunk_indices):
        delayed_apply_mapping = joblib.delayed(self.apply_mapping)
        jobs = [delayed_apply_mapping(downseg[idx], lut=lut, zarr_store=zout_down, slice_idx=idx) for idx in chunk_indices]
        # Using downseg array here as input arr to be re-mapped? & writes out to zout_down? 
        
        for job in jobs: 
            print("Remapping Job:", job)
        
        executor = joblib.Parallel(n_jobs=-1, backend='threading')
        executor(jobs)
        return


    # --------- Helper Methods ---------
    def _reconcile_labels(self, seg_array, zout_tmp):
        current_labels = da.from_zarr(seg_array)
        unique_labels = da.unique(current_labels).compute()
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:] 

        # 2. Get the number of classes we need, from self.maximum_objects_per_class
        num_classes = math.ceil(len(unique_labels)/self.maximum_objects_per_class)
        # Turn this into an int array
        new_ids = []

        for class_id in range(1, num_classes+1):
            min_id = (class_id*self.maximum_objects_per_class) + 1
            max_id = ((class_id+1) * self.maximum_objects_per_class) - 1

            if max_id > len(unique_labels):
                max_obj_id = len(unique_labels)+1 % max(num_classes-1, 1) 
                max_id = max_obj_id + (class_id*self.maximum_objects_per_class)

            new_ids.extend(np.arange(min_id, max_id))

        # 3. Unique_labels is already sorted, as is new_ids
        id_map = dict(zip(unique_labels, new_ids))

        # Build a global LookUp Table:
        max_key = max(unique_labels)
        lut = np.arange(max_key+1, dtype=np.int64)
        for old, new in id_map.items():
            lut[old] = new
        
        # 4. Apply the dict map in parallel, to each chunk in seg array 
        # use the outseg zarr store (zout_down) & original chunk_indices
        delayed_apply_mapping = joblib.delayed(self.apply_mapping)
        jobs = [delayed_apply_mapping(current_labels[idx], lut=lut, zarr_store=zout_tmp, slice_idx=idx) for idx in chunk_indices]
        # Using downseg array here as input arr to be re-mapped? & writes out to zout_down? 
        for job in jobs: print("Remapping Job:", job)
        executor = joblib.Parallel(n_jobs=-1, backend='threading')
        executor(jobs) 
        '''Label reconciliation Done!'''

        return

    def _get_strips(self, chunk_sizes:list, padding:int):

        # This first gets slice objects for horizontal + vertical chunk boundaries
        # We need to know: a list of chunk indices
            # i.e. where they start and end
        # Horizontal chunks will be computed from the x coords, vertical from the y
        # Then return horizontal and vertical strips = dask image arrays 

        y_chunk = image_down.shape[0]//2
        x_chunk = image_down.shape[1]//2

        v_idx = (slice(0, image_down.shape[0], None), slice(x_chunk-padding, x_chunk+padding, None))
        h_idx = (slice(y_chunk-padding, y_chunk+padding, None), slice(0, image_down.shape[1], None))
        vertical_strip = image_down[v_idx]
        horizontal_strip = image_down[h_idx]

        return {'vertical': ['vertical srips'], 'horizontal': ['horizontal strips']}

    def process_segmentation_chunk(self, engine, image, axis, plane, y, x, zarr_store, slice_idx, merge_labels=False):
        if isinstance(image, da.Array):
            input_array = input_array.compute() 

        # Get the seg result:
        seg, axis, plane, y, x = self._get_segmentation(engine, image, axis, plane, y, x)

        # Deal with label updating within zarr_store:
        # Get this chunk's classID:
        id = (slice_idx[0].start + slice_idx[1].stop)//100 * 100
        class_id = self.class_ids[id]
        old_divisor = self.maximum_objects_per_class

        unique_labels = np.unique(seg[seg>0])

        # Update all labels in seg to be class_id
        mapping = {
            old_label: (old_label-old_divisor)+class_id
            for old_label in unique_labels
        }
        seg = self.apply_mapping(seg, mapping)

        if not merge_labels:
            final = seg

        else:
            print("merging labels...")
            # Merge overlapping labels into same label
            existing = zarr_store[slice_idx]

            seg2_to_seg1 = self.build_mapping(existing, seg)
            seg2_fixed = self.apply_mapping(seg, seg2_to_seg1)

            final = existing.copy()
            mask = seg2_fixed > 0
            final[mask] = seg2_fixed[mask]
            print("Saving...")

        zarr_store[slice_idx] = final
        return


    def _downsample_array(self, image, scale):
        # This should also take the scale factor/resolution level to downsample to
        return image[::scale, ::scale]
    
    def _create_class_ids(self, chunk_indices):
        class_id = 1
        divisor = 1000

        for idx in chunk_indices:
            id = (idx[0].start + idx[1].stop)//100 * 100
            self.class_ids[id] = class_id*divisor
            class_id += 1

        ## Need to handle the dict of vertical + hori strips/indices:
        for chunk_idx in [v_idx, h_idx]:
            id = (chunk_idx[0].start + chunk_idx[1].stop)//100 * 100
            self.class_ids[id] = class_id*divisor
            class_id += 1
        
        return
    
# ----- Label Postprocessing -----

    def apply_mapping(self, seg, mapping=None, lut=None, zarr_store=None, slice_idx=None):

        print("replacing labels...")

        if lut is None:
            max_key = max(mapping)
            lut = np.arange(max_key + 1, dtype=np.int64)

            for old, new in mapping.items():
                lut[old] = new

        out = lut[seg]

        # print("MAP: ", mapping)
        # print("Unique seg:", np.unique(seg))
        # print("OUT>0 LABELS: ", out[out>0])
        
        # out = seg.copy()    
        # for old, new in mapping.items():
        #     out[seg == old] = new   

        if zarr_store and slice_idx:
            zarr_store[slice_idx] = out
            print("exported finalised labels!")
            return
        
        return out

    def build_mapping(self, seg1, seg2, min_conf=0.8):
        mapping = {}
        seg2_labels = np.unique(seg2[seg2>0])

        for l2 in seg2_labels:
            mask = seg2 == l2

            overlap = seg1[mask]
            overlap = overlap[overlap > 0]

            if len(overlap) == 0:
                continue

            labels, counts = np.unique(overlap, return_counts=True)

            best = np.argmax(counts)
            best_label = labels[best]

            conf = counts[best] / counts.sum()

            if conf >= min_conf:
                mapping[l2] = best_label

        return mapping

    def align_seg2_to_seg1(self, seg1, seg2, min_overlap=0.5):
        """
        For each seg2 label:
            assign it the seg1 label it overlaps most
        """

        seg2_out = np.zeros_like(seg2)

        seg2_labels = np.unique(seg2)
        seg2_labels = seg2_labels[seg2_labels > 0]

        for l2 in seg2_labels:

            mask = seg2 == l2

            overlap = seg1[mask]
            overlap = overlap[overlap > 0]

            if len(overlap) == 0:
                continue

            labels, counts = np.unique(overlap, return_counts=True)

            best_idx = np.argmax(counts)
            best_label = labels[best_idx]

            confidence = counts[best_idx] / counts.sum()

            if confidence >= min_overlap:
                seg2_out[mask] = best_label

        return seg2_out