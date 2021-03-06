# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2020, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import multiprocessing
import os

from typing import Any, Dict

GT4PY_INSTALLATION_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths (taken from user's environment vars when possible)
BOOST_ROOT: str = os.environ.get(
    "BOOST_ROOT", os.environ.get("BOOST_HOME", os.path.abspath("/usr/local"))
)

CUDA_ROOT: str = os.environ.get(
    "CUDA_HOME", os.environ.get("CUDA_PATH", os.path.abspath("/usr/local/cuda"))
)

GT_REPO_PATH: str = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_external_src", "gridtools")
)

GT_INCLUDE_PATH: str = os.path.abspath(os.path.join(GT_REPO_PATH, "include"))

# Settings dict
build_settings: Dict[str, Any] = {
    "boost_include_path": os.path.join(BOOST_ROOT, "include"),
    "cuda_bin_path": os.path.join(CUDA_ROOT, "bin"),
    "cuda_include_path": os.path.join(CUDA_ROOT, "include"),
    "cuda_library_path": os.path.join(CUDA_ROOT, "lib64"),
    "cuda_arch": os.environ.get("CUDA_ARCH", None),
    "gt_include_path": os.environ.get("GT_INCLUDE_PATH", GT_INCLUDE_PATH),
    "extra_compile_args": {"cxx": [], "nvcc": []},
    "extra_link_args": [],
    "parallel_jobs": multiprocessing.cpu_count(),
}

cache_settings: Dict[str, Any] = {
    "dir_name": os.environ.get("GT_CACHE_DIR_NAME", ".gt_cache"),
    "root_path": os.environ.get("GT_CACHE_ROOT", os.path.abspath(".")),
}

code_settings: Dict[str, Any] = {"root_package_name": "_GT_"}
