# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
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

import abc
import hashlib
import inspect
import numbers
import os
import types

import jinja2
import numpy as np

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as ir_utils

from gt4py import backend as gt_backend
from gt4py import config as gt_config
from gt4py import definitions as gt_definitions
from gt4py import ir as gt_ir
from gt4py.utils import text as gt_text
from .base_gt_backend import BaseGTBackend
from . import pyext_builder


def make_x86_layout_map(mask):
    ctr = iter(range(sum(mask)))
    if len(mask) < 3:
        layout = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask = [*mask[3:], *mask[:3]]
        layout = [next(ctr) if m else None for m in swapped_mask]

        layout = [*layout[-3:], *layout[:-3]]

    return tuple(layout)


def x86_is_compatible_layout(field):
    stride = 0
    layout_map = make_x86_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


def make_mc_layout_map(mask):
    ctr = reversed(range(sum(mask)))
    if len(mask) < 3:
        layout = [next(ctr) if m else None for m in mask]
    else:
        swapped_mask = list(mask)
        tmp = swapped_mask[1]
        swapped_mask[1] = swapped_mask[2]
        swapped_mask[2] = tmp

        layout = [next(ctr) if m else None for m in swapped_mask]

        tmp = layout[1]
        layout[1] = layout[2]
        layout[2] = tmp

    return tuple(layout)


def mc_is_compatible_layout(field):
    stride = 0
    layout_map = make_mc_layout_map(field.mask)
    if len(field.strides) < len(layout_map):
        return False
    for dim in reversed(np.argsort(layout_map)):
        if field.strides[dim] < stride:
            return False
        stride = field.strides[dim]
    return True


class DawnBaseGenerator(abc.ABC):

    SOURCE_LINE_LENGTH = 120
    TEMPLATE_INDENT_SIZE = 4
    DOMAIN_ARG_NAME = "_domain_"
    ORIGIN_ARG_NAME = "_origin_"
    SPLITTERS_NAME = "_splitters_"

    TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "stencil_module.py.in")

    def __init__(self, backend_class, options):
        assert issubclass(backend_class, gt_backend.BaseBackend)
        self.backend_class = backend_class
        self.options = types.SimpleNamespace(**options)
        with open(self.TEMPLATE_PATH, "r") as f:
            self.template = jinja2.Template(f.read())

        self.stencil_id = None

    def __call__(
        self,
        stencil_id,
        domain_info: gt_definitions.DomainInfo,
        field_info: dict,
        parameter_info: dict,
        api_signature: list,
        sources: dict,
        externals: dict,
    ):
        self.stencil_id = stencil_id
        self.domain_info = domain_info
        self.field_info = field_info
        self.parameter_info = parameter_info
        self.api_signature = api_signature
        self.sources = sources or {}
        self.externals = externals or {}

        field_names = [name for name in field_info.keys()]
        fields_dict_call = "dict({})".format(", ".join(f"{name}={name}" for name in field_names))
        params_dict_call = "dict({})".format(
            ", ".join(f"{name}={name}" for name in parameter_info.keys())
        )

        domain_info_str = repr(domain_info)
        field_info_str = repr(field_info)
        parameter_info_str = repr(parameter_info)

        gt_source = {
            key: gt_text.format_source(value, line_length=self.SOURCE_LINE_LENGTH)
            for key, value in self.sources
        }

        gt_constants = {}
        for name, value in self.externals.items():
            assert isinstance(value, numbers.Number)
            gt_constants[name] = repr(value)

        gt_options = dict(self.options.__dict__)
        if "build_info" in gt_options:
            del gt_options["build_info"]

        module_source = self.template.render(
            imports=self.imports_source,
            module_members=self.module_members_source,
            class_name=self.stencil_class_name,
            class_members=self.class_members_source,
            gt_backend=self.backend_name,
            gt_source=gt_source,
            gt_domain_info=domain_info_str,
            gt_field_info=field_info_str,
            gt_parameter_info=parameter_info_str,
            gt_constants=gt_constants,
            gt_options=gt_options,
            stencil_signature=self.signature,
            fields_dict_call=fields_dict_call,
            params_dict_call=params_dict_call,
            synchronization=self.synchronization_source,
            mark_modified=self.mark_modified_source,
            implementation=self.implementation_source,
        )
        module_source = gt_text.format_source(module_source, line_length=self.SOURCE_LINE_LENGTH)

        return module_source

    @property
    def backend_name(self):
        return self.backend_class.name

    @property
    def stencil_class_name(self):
        return self.backend_class.get_stencil_class_name(self.stencil_id)

    @property
    def synchronization_source(self):
        return ""

    @property
    def mark_modified_source(self):
        return ""

    @property
    def signature(self):
        args = []
        keyword_args = ["*"]
        for arg in self.api_signature:
            if arg.is_keyword:
                if arg.default is not gt_ir.Empty:
                    keyword_args.append(
                        "{name}={default}".format(name=arg.name, default=arg.default)
                    )
                else:
                    keyword_args.append(arg.name)
            else:
                if arg.default is not gt_ir.Empty:
                    args.append("{name}={default}".format(name=arg.name, default=arg.default))
                else:
                    args.append(arg.name)

        if len(keyword_args) > 1:
            args.extend(keyword_args)
        signature = ", ".join(args)

        return signature

    @property
    def imports_source(self):
        return ""

    @property
    def module_members_source(self):
        return ""

    @property
    def class_members_source(self):
        return ""

    @property
    @abc.abstractmethod
    def implementation_source(self):
        pass


class DawnPythonGenerator(DawnBaseGenerator):
    def __init__(self, backend_class, options):
        super().__init__(backend_class, options)

    @property
    def imports_source(self):
        source = """
import functools

from gt4py import utils as gt_utils

pyext_module = gt_utils.make_module_from_file("{pyext_module_name}", "{pyext_file_path}", public_import=True)
        """.format(
            pyext_module_name=self.options.pyext_module_name,
            pyext_file_path=self.options.pyext_file_path,
        )

        return source

    @property
    def implementation_source(self):
        sources = gt_text.TextBlock(indent_size=gt_backend.BaseGenerator.TEMPLATE_INDENT_SIZE)

        args = []
        for arg in self.api_signature:
            args.append(arg.name)
            if arg.name in self.field_info:
                args.append("list(_origin_['{}'])".format(arg.name))

        source = """
# Load or generate a GTComputation object for the current domain size
pyext_module.run_computation(list(_domain_), {run_args}, exec_info)
""".format(
            run_args=", ".join(args)
        )
        #         if self.backend_name == "gtcuda":
        #             source = (
        #                 source
        #                 + """import cupy
        # cupy.cuda.Device(0).synchronize()
        # """
        #             )
        source = source + (
            """if exec_info is not None:
    exec_info["run_end_time"] = time.perf_counter()
"""
        )
        sources.extend(source.splitlines())

        return sources.text


class BaseDawnBackend(gt_backend.BaseBackend):

    GENERATOR_CLASS = DawnPythonGenerator

    GT_BACKEND_OPTS = {
        "verbose": {"versioning": False},
        "clean": {"versioning": False},
        "gtcache_size": {"versioning": True},
        "debug_mode": {"versioning": True},
        "add_profile_info": {"versioning": True},
    }

    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    TEMPLATE_FILES = {
        "computation.hpp": "computation.hpp.in",
        "computation.src": "dawn_computation.src.in",
        "bindings.cpp": "bindings.cpp.in",
    }

    @classmethod
    def get_pyext_module_name(cls, stencil_id, *, qualified=False):
        module_name = cls.get_stencil_module_name(stencil_id, qualified=qualified) + "_pyext"
        return module_name

    @classmethod
    def get_pyext_class_name(cls, stencil_id):
        module_name = cls.get_stencil_class_name(stencil_id) + "_pyext"
        return module_name

    @classmethod
    def get_pyext_build_path(cls, stencil_id):
        path = os.path.join(
            cls.get_stencil_package_path(stencil_id),
            cls.get_pyext_module_name(stencil_id) + "_BUILD",
        )

        return path

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options):
        raise NotImplementedError(
            "'generate_extension()' method must be implemented by subclasses"
        )

    @classmethod
    def generate_cache_info(cls, stencil_id, extra_cache_info):
        cache_info = super(BaseDawnBackend, cls).generate_cache_info(stencil_id, {})

        cache_info["pyext_file_path"] = extra_cache_info["pyext_file_path"]
        cache_info["pyext_md5"] = hashlib.md5(
            open(cache_info["pyext_file_path"], "rb").read()
        ).hexdigest()

        return cache_info

    @classmethod
    def validate_cache_info(cls, stencil_id, cache_info):
        try:
            assert super(BaseDawnBackend, cls).validate_cache_info(stencil_id, cache_info)
            pyext_md5 = hashlib.md5(open(cache_info["pyext_file_path"], "rb").read()).hexdigest()
            result = pyext_md5 == cache_info["pyext_md5"]

        except Exception:
            result = False

        return result

    @classmethod
    def generate(cls, stencil_id, definition_ir, definition_func, options):
        cls._check_options(options)

        # Generate the Python binary extension (checking if GridTools sources are installed)
        if not os.path.isfile(
            os.path.join(gt_config.GT_INCLUDE_PATH, "gridtools", "common", "defs.hpp")
        ):
            raise RuntimeError(
                "Missing GridTools sources. Run 'python setup.py install_gt_sources'."
            )
        pyext_module_name, pyext_file_path = cls.generate_extension(
            stencil_id, definition_ir, options
        )

        # Generate and return the Python wrapper class
        generator_options = options.as_dict()
        generator_options["pyext_module_name"] = pyext_module_name
        generator_options["pyext_file_path"] = pyext_file_path

        extra_cache_info = {"pyext_file_path": pyext_file_path}

        return cls._build(
            stencil_id, definition_ir, definition_func, generator_options, extra_cache_info
        )

    @classmethod
    def generate_extension(cls, stencil_id, definition_ir, options):

        interval = ir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

        # create the out = in[i+1] statement
        body_ast = ir_utils.make_ast(
            [
                ir_utils.make_assignment_stmt(
                    ir_utils.make_field_access_expr("out", [0, 0, 0]),
                    ir_utils.make_field_access_expr("in", [1, 0, 0]),
                    "=",
                )
            ]
        )

        vertical_region_stmt = ir_utils.make_vertical_region_decl_stmt(
            body_ast, interval, SIR.VerticalRegion.Forward
        )

        stencil_short_name = stencil_id.qualified_name.split(".")[-1]
        sir = ir_utils.make_sir(
            "copy_stencil.cpp",
            [
                ir_utils.make_stencil(
                    stencil_short_name,
                    ir_utils.make_ast([vertical_region_stmt]),
                    [ir_utils.make_field("in"), ir_utils.make_field("out")],
                )
            ],
        )

        # Generate sources
        source = dawn4py.compile_to_source(sir)
        stencil_unique_name = cls.get_pyext_class_name(stencil_id)
        module_name = cls.get_pyext_module_name(stencil_id)
        pyext_sources = {f"_dawn_{stencil_short_name}.hpp": source}
        dawn_backend = "gt"
        gt_backend = "x86"

        arg_fields = [
            {"name": name, "dtype": "double", "layout_id": i}
            for i, name in enumerate(["in", "out"])
        ]
        header_file = "computation.hpp"
        parameters = []

        template_args = dict(
            arg_fields=arg_fields,
            dawn_backend=dawn_backend,
            gt_backend=gt_backend,
            header_file=header_file,
            module_name=module_name,
            parameters=parameters,
            stencil_short_name=stencil_short_name,
            stencil_unique_name=stencil_unique_name,
        )

        for key, file_name in cls.TEMPLATE_FILES.items():
            with open(os.path.join(cls.TEMPLATE_DIR, file_name), "r") as f:
                template = jinja2.Template(f.read())
                pyext_sources[key] = template.render(**template_args)

        # Build extension module
        pyext_build_path = os.path.relpath(cls.get_pyext_build_path(stencil_id))
        os.makedirs(pyext_build_path, exist_ok=True)
        sources = []
        for key, source in pyext_sources.items():
            src_file_name = os.path.join(pyext_build_path, key)
            src_ext = src_file_name.split(".")[-1]
            if src_ext != "hpp":
                if src_ext == "src":
                    src_file_name = src_file_name.replace("src", "cpp")
                sources.append(src_file_name)

            with open(src_file_name, "w") as f:
                f.write(source)

        # Build extension module
        pyext_opts = dict(
            verbose=options.backend_opts.pop("verbose", True),
            clean=options.backend_opts.pop("clean", False),
            debug_mode=options.backend_opts.pop("debug_mode", False),
            add_profile_info=options.backend_opts.pop("add_profile_info", False),
        )

        include_dirs = [
            "{install_dir}/_external_src".format(
                install_dir=os.path.dirname(inspect.getabsfile(dawn4py))
            )
        ]

        pyext_target_path = cls.get_stencil_package_path(stencil_id)
        qualified_pyext_name = cls.get_pyext_module_name(stencil_id, qualified=True)
        module_name, file_path = pyext_builder.build_gtcpu_ext(
            qualified_pyext_name,
            sources=sources,
            build_path=pyext_build_path,
            target_path=pyext_target_path,
            extra_include_dirs=include_dirs,
            **pyext_opts,
        )
        assert module_name == qualified_pyext_name

        return module_name, file_path

    @classmethod
    def _build(
        cls, stencil_id, definition_ir, definition_func, generator_options, extra_cache_info
    ):

        generator = cls.GENERATOR_CLASS(cls, options=generator_options)

        parallel_axes = definition_ir.domain.parallel_axes or []
        sequential_axis = definition_ir.domain.sequential_axis.name
        domain_info = gt_definitions.DomainInfo(
            parallel_axes=tuple(ax.name for ax in parallel_axes),
            sequential_axis=sequential_axis,
            ndims=len(parallel_axes) + (1 if sequential_axis else 0),
        )

        field_info = {}
        parameter_info = {}
        fields = {item.name: item for item in definition_ir.api_fields}
        parameters = {item.name: item for item in definition_ir.parameters}
        halo_size = generator_options["backend_opts"].get(
            "max_halo_points", dawn4py.Options().max_halo_points
        )
        boundary = gt_definitions.Boundary(
            [(halo_size, halo_size) for _ in domain_info.parallel_axes] + [(0, 0)]
        )
        for arg in definition_ir.api_signature:
            if arg.name in fields:
                field_info[arg.name] = gt_definitions.FieldInfo(
                    access=gt_definitions.AccessKind.READ_WRITE,
                    dtype=fields[arg.name].data_type.dtype,
                    boundary=boundary,
                )
            else:
                parameter_info[arg.name] = gt_definitions.ParameterInfo(
                    dtype=parameters[arg.name].data_type.dtype
                )

        module_source = generator(
            stencil_id,
            domain_info,
            field_info,
            parameter_info,
            definition_ir.api_signature,
            definition_ir.sources,
            definition_ir.externals,
        )

        file_name = cls.get_stencil_module_path(stencil_id)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as f:
            f.write(module_source)
        cls.update_cache(stencil_id, extra_cache_info)

        return cls._load(stencil_id, definition_func)


class DawnCPUBackend(BaseDawnBackend):
    pass


@gt_backend.register
class DawnGTX86Backend(DawnCPUBackend):

    name = "dawn:gtx86"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": make_x86_layout_map,
        "is_compatible_layout": x86_is_compatible_layout,
    }

    _CPU_ARCHITECTURE = "x86"


@gt_backend.register
class DawnGTMCBackend(DawnCPUBackend):

    name = "dawn:gtmc"
    options = BaseGTBackend.GT_BACKEND_OPTS
    storage_info = {
        "alignment": 8,
        "device": "cpu",
        "layout_map": make_mc_layout_map,
        "is_compatible_layout": mc_is_compatible_layout,
    }

    _CPU_ARCHITECTURE = "mc"
