import math
import zarr
import numba
import numpy as np

from multiprocessing import Pool
from skimage.transform import resize
from ome_zarr.writer import write_multiscale, write_multiscales_metadata
from empanada.array_utils import rle_to_ranges

__all__ = [
    'zarr_fill_instances'
]

@numba.jit(nopython=True)
def chunk_ranges(ranges, modulo, divisor):
    # shift the ranges based on a divisor
    # this will give us the chunk indices
    # for starts and ends of each range
    ci_s = (ranges[:, 0] % modulo) // divisor 
    # ranges exclude the last index here
    ci_e = ((ranges[:, 1] - 1) % modulo) // divisor

    # check where chunk indices are not
    # the same between start and end
    chunked_ranges = []
    for cs, ce, r in zip(ci_s, ci_e, ranges):
        # split or keep the range
        if cs != ce or (r[1] - r[0] > divisor):
            # convert from a range to a list of indices
            # add one because ranges are non-inclusive endpoints
            range_indices = np.arange(r[0], r[1] + 1, dtype=ranges.dtype)
            cr_indices = [(ri % modulo) // divisor for ri in range_indices]
            split_idx = [0]
            last_c = cr_indices[0]
            for i,c in enumerate(cr_indices[1:], 1):
                if c != last_c:
                    split_idx.append(i)
                    last_c = c
                
            # append last index to create consistent ranges
            if split_idx[-1] != len(range_indices) - 1:
                split_idx.append(-1)
                
            for i, j in zip(split_idx[:-1], split_idx[1:]):
                chunked_ranges.append([range_indices[i], range_indices[j]])
        else:
            # no need to split the range
            chunked_ranges.append([r[0], r[1]])
    
    return chunked_ranges

@numba.jit(nopython=True)
def fill_func(seg1d, coords, instance_id):
    r"""Fills coords in seg1d (raveled image) with value instance_id"""
    # inplace fill seg1d with instance_id
    # at the given xy raveled coords
    for coord in coords:
        s, e = coord
        seg1d[s:e] = instance_id

    return seg1d

def fill_zarr_mp(*args):
    r"""Helper function for multiprocessing the filling of zarr slices"""
    # fills zarr array with multiprocessing
    slices, instances, array = args[0]
    z, y, x = [sl.start for sl in slices]
    
    seg = array[slices]
    chunk_shape = seg.shape
    
    seg = seg.reshape(-1)
    for instance_id, ranges in instances.items():
        zs, ys, xs = np.unravel_index(ranges[:, 0], array.shape)
        ze, ye, xe = np.unravel_index(ranges[:, 1] - 1, array.shape)

        zs -= z
        ze -= z
        ys -= y
        ye -= y
        xs -= x
        xe -= x

        starts = np.ravel_multi_index((zs, ys, xs), chunk_shape)
        ends = np.ravel_multi_index((ze, ye, xe), chunk_shape)
        
        fill_func(seg, np.stack([starts, ends + 1], axis=1), int(instance_id))

    array[slices] = seg.reshape(chunk_shape)

def zarr_fill_instances(array, instances, processes=4):
    r"""Fills a zarr array in-place with instances.

    Args:
        array: zarr.Array of size (d, h, w)

        instances: Dictionary. Keys are instance_ids (integers) and
            values are another dictionary containing the run length
            encoding (keys: 'starts', 'runs').

        processes: Integer, the number of processes to run.

    """
    # make sure number of processes doesn't exceed chunks
    processes = min(array.nchunks, processes)
    
    d, h, w = array.shape
    dc, hc, wc = array.chunks
    cd, ch, cw = math.ceil(d / dc), math.ceil(h / hc), math.ceil(w / wc)
    
    # enumerate and define ROIs for each chunk
    chunks = {}
    for i, z1 in enumerate(range(0, d, dc)):
        for j, y1 in enumerate(range(0, h, hc)):
            for k, x1 in enumerate(range(0, w, wc)):
                z2 = min(d, z1 + dc)
                y2 = min(h, y1 + hc)
                x2 = min(w, x1 + wc)

                zslice = slice(z1, z2)
                yslice = slice(y1, y2)
                xslice = slice(x1, x2)
                
                chunk_idx = (i * ch * cw) + (j * cw) + k
                
                chunks[chunk_idx] = {
                    'slices': (zslice, yslice, xslice),
                    'instances': {}
                }
    
    chunked_instances = {}
    for instance_id, instance_attrs in instances.items():
        # convert from runs to ranges
        ranges = rle_to_ranges(np.stack(
            [instance_attrs['starts'], instance_attrs['runs']], axis=1
        ))
        
        zmodulo = d * h * w
        zdivisor = dc * h * w
        ranges = np.array(chunk_ranges(ranges, zmodulo, zdivisor))
                          
        ymodulo = h * w
        ydivisor = hc * w
        ranges = np.array(chunk_ranges(ranges, ymodulo, ydivisor))
                          
        xmodulo = w
        xdivisor = wc
        ranges = np.array(chunk_ranges(ranges, xmodulo, xdivisor))
        
        # get chunk_ids for each range
        zc = (ranges[:, 0] % zmodulo) // zdivisor
        yc = (ranges[:, 0] % ymodulo) // ydivisor
        xc = (ranges[:, 0] % xmodulo) // xdivisor
        
        chunk_indices = (zc * ch * cw) + (yc * cw) + xc 
        
        # group ranges by the chunk in which they reside
        sort_idx = np.argsort(chunk_indices)
        ranges = ranges[sort_idx]
        chunk_indices = chunk_indices[sort_idx]
        unq_chunks, unq_idx = np.unique(chunk_indices, return_index=True)
        chunked_ranges = np.split(ranges, unq_idx[1:])
        
        for chunk_id, cranges in zip(unq_chunks, chunked_ranges):
            chunks[chunk_id]['instances'][instance_id] = cranges 
            
    # setup for multiprocessing
    n = len(chunks)
    arg_iter = zip(
        [cdict['slices'] for cdict in chunks.values()],
        [cdict['instances'] for cdict in chunks.values()],
        [array] * n
    )
    
    # fill the zarr volume. nothing to
    # return because it's done inplace
    with Pool(processes) as pool:
        pool.map(fill_zarr_mp, arg_iter)



# ---------------- OME-Zarr Helper Functions ----------------

def _generate_tiles(shape, tile_shape):
    '''Generates Slice indices for each chunk, given a tuple of chunk(or tile)_shapes'''

    grids = [range(0, s, t) for s, t in zip(shape, tile_shape)]

    for coords in np.ndindex(*[len(g) for g in grids]):
        yield tuple(
            slice(grids[d][coords[d]],
                  min(grids[d][coords[d]] + tile_shape[d], shape[d]))
            for d in range(len(shape))
        )

def _write_empty_chunk(store_path, image, inp_scale=None, inp_units=None):
    '''Write a single empty array to OME-Zarr
        Returns: da.array|zarr.array '''
    # To-Do: Minimum metadata needed is datasets and axes dicts
    # Make these parameters, use defaults if none given
    # Rename to _write_empty_label_array (NOT writing chunk)
    # Input params should be: desired shape, desired chunk sizes, axes, datasets & name

    ndims = image.ndim
    dim_names = ["z", "y", "x"][-1*ndims:]
    if inp_scale is None:
        inp_scale = [1]*ndims
    if inp_units is None:
        inp_units = ["micrometer"]*ndims
    else:
        inp_units = [str(u) for u in inp_units]
        inp_scale = list(inp_scale)

    group_name = "tmp"
    array_name = "s0"

    datasets=[
        {"path": array_name, "coordinateTransformations": 
         [{"type": "scale", "scale": [n*2 for n in inp_scale]}]}
            ]
    axes=[{"name": nm, "type": "space", "unit": unit} 
          for unit, nm in zip(inp_units, dim_names)]

    root = zarr.open_group(store_path, mode="w", zarr_format=3)
    
    root.attrs["labels"] = [group_name]
    labels_root = root.require_group("labels")
    label_group = labels_root.require_group(group_name)

    write_multiscales_metadata(
        label_group,
        datasets=datasets,
        axes=axes,
        type="labels"
    )

    down_shape = tuple(s // 2 for s in image.shape)
    down_chunks = tuple(max(1, c // 2) for c in down_shape)

    z1 = label_group.require_array(
                    array_name,
                    shape=down_shape,
                    chunks=down_chunks,
                    dtype="int32",
                    dimension_names=dim_names,
                    )

    print(f"Initial empty labels array written at: {store_path}/labels/{group_name}/{array_name}")
    return z1


def _write_multiscale(store_path, full_seg, scale_factors, multidims, datasets, axes, name="labels"):
    # Rename to _write_label_multiscale

    root = zarr.open_group(store_path, mode="a", zarr_format=3)
    seg_group = root.require_group("labels/seg")
    
    print("WRITING METADATA:", scale_factors, "\n", axes, "\n", datasets)

    write_label_pyramid_from_image(full_seg, seg_group, 
                                   scale_factors, multidims, axes, datasets, name) 
    

    # write_multiscales_metadata(group=seg_group, datasets=datasets, 
                                                # axes=axes, name="labels")
    
    return


def write_label_pyramid_from_image(labels, group, scale_factors, multidims, axes, coordinate_transformations, name):
    # 1. take scale factors, use to get the array dir names
    path_nms = list(range(len(scale_factors)+1))

    # 2. Iterate over the target shapes:
    pyramid = []
    for dims in multidims: # Create pyramid of dask Arrays
        scaled_arr = resize(labels, dims, order=0, 
                        mode='reflect', anti_aliasing=False, preserve_range=True)
        pyramid.append(scaled_arr)
        
    # Write out the pyramid to the /labels/seg group
    write_multiscale(pyramid, group, axes, coordinate_transformations, name)

    # Add "image-label" key to "ome"
    ome = dict(group.attrs.get("ome", {}))
    ome["image-label"] = {"version": "0.5"}
    group.attrs["ome"] = ome

    return

