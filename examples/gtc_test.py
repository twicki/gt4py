#!/usr/bin/env python

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None

import os
import sys

import numpy as np

# from fv3core.stencils.delnflux import copy_corners_x_nord
# from fv3core.stencils.delnflux import fx_calc_stencil_column
# from fv3core.stencils.delnflux import d2_highorder_stencil
# from fv3core.stencils.xppm import compute_x_flux
# from fv3core.stencils.yppm import compute_y_flux
# from fv3core.stencils.xtp_u import _xtp_u
# from fv3core.stencils.fv_subgridz import init
from fv3core.stencils.ytp_v import _ytp_v

from fv3core.utils.mpi import MPI
from fv3core.utils.typing import FloatField, FloatFieldIJ

import gt4py as gt
import gt4py.storage as gt_storage
from gt4py.gtscript import (
    FORWARD,
    IJ,
    IJK,
    PARALLEL,
    Field,
    I,
    J,
    K,
    computation,
    horizontal,
    interval,
    region,
    stencil,
)


# gt_backend = "gtcuda"  # "gtx86"
gt_backend = "gtc:cuda"
# gt_backend = "gtc:gt:gpu"
# np_backend = "numpy"
np_backend = "gtx86"


def mask_from_shape(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (False, False, True)
    return (True,) * len(shape) + (False,) * (3 - len(shape))


def arrays_to_storages(array_dict: dict, backend: str, origin: tuple) -> tuple:
    return {
        name: gt_storage.from_array(
            data=array,
            backend=backend,
            default_origin=origin,
            shape=array.shape,
            mask=mask_from_shape(array.shape),
            managed_memory=True,
        )
        for name, array in array_dict.items()
    }


def double_data(input_data):
    output_data = {}
    for name, array in input_data.items():
        array2 = np.concatenate((array, array), axis=1)
        array3 = np.concatenate((array2, array2), axis=0)
        output_data[name] = array3
    return output_data


def main():
    hsize = 18
    vsize = hsize + 1
    nhalo = 3  # 2  # 1

    definition_func = _ytp_v

    data_file_prefix = sys.argv[1] if len(sys.argv) > 1 else definition_func.__name__
    is_parallel: bool = MPI is not None and MPI.COMM_WORLD.Get_size() > 1
    data_file_prefix += f"_r{MPI.COMM_WORLD.Get_rank()}" if is_parallel else ""
    input_data = dict(np.load(f"{data_file_prefix}.npz", allow_pickle=True))

    origin = tuple(input_data.pop("origin", []))
    if not any(origin):
        origin = (0, 0, 0)

    domain = tuple(input_data.pop("domain", []))
    if not any(domain):
        domain = (hsize, hsize, vsize)

    externals = input_data.pop("externals", {})
    if externals:
        externals = externals[()]
        axes = {"I": I, "J": J}
        for name, value in externals.items():
            if isinstance(value, dict) and "axis" in value:
                axis_index = axes[value["axis"]][value["index"]] + value["offset"]
                assert axis_index.__dict__ == value
                externals[name] = axis_index

    # GreedyMerging causes the illegal memory errors...
    do_rebuild: bool = True
    gt_stencil = stencil(
        definition=definition_func,
        backend=gt_backend,
        externals=externals,
        rebuild=do_rebuild,
    )
    np_stencil = stencil(
        definition=definition_func,
        backend=np_backend,
        externals=externals,
        rebuild=False,
    )

    n_doubles: int = 0
    if n_doubles:
        for _ in range(n_doubles):
            input_data = double_data(input_data)
            domain = (domain[0] * 2, domain[1] * 2, domain[2])

    gt_storages = arrays_to_storages(input_data, gt_backend, origin)
    np_storages = arrays_to_storages(input_data, np_backend, origin)

    n_runs = 9
    total_time = 0.0
    for _ in range(n_runs):
        exec_info = {}
        gt_stencil(domain=domain, origin=origin, exec_info=exec_info, **gt_storages)
        cpp_run_time = (
            exec_info["run_cpp_end_time"] - exec_info["run_cpp_start_time"]
        ) * 1e3
        total_time += cpp_run_time
    mean_time = total_time / float(n_runs)
    print(f"mean_time (backend={gt_backend}, domain={domain}) = {mean_time}")

    np_stencil(domain=domain, origin=origin, **np_storages)

    fail_arrays = {}
    for name in gt_storages.keys():
        gt_array = np.asarray(gt_storages[name])
        np_array = np.asarray(np_storages[name])
        if not np.allclose(gt_array, np_array, equal_nan=True):
            fail_arrays[name] = (gt_array, np_array)

    if fail_arrays:
        for array_tuple in fail_arrays.values():
            gt_array, np_array = array_tuple
            diff_array = gt_array - np_array
            diff_indices = np.transpose(diff_array[:, :, 0].nonzero())
            fail_ratio = diff_indices.shape[0] / (
                diff_array.shape[0] * diff_array.shape[1]
            )

            # print(f"np_array = {np_array[:, :, 0]}")
            # print(f"gt_array = {gt_array[:, :, 0]}")
            # print(f"diff_array = {diff_array[:, :, 0]}")
            # print(f"diff_indices = {diff_indices}")
            print(f"fail_ratio = {fail_ratio}")
        print(f"{fail_arrays.keys()} fail")
    else:
        print("All good!")


if __name__ == "__main__":
    main()
