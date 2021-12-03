# -*- coding: utf-8 -*-
#
# GTC Toolchain - GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from eve import NodeVisitor
from eve.concepts import TreeNode
from eve.utils import XIterable, xiter
from gtc import common, oir


@dataclass(frozen=True)
class Access:
    field: str
    offset: Tuple[int, int, int]
    is_write: bool
    in_mask: bool = False
    region: Optional[oir.HorizontalMask] = None

    @property
    def is_read(self) -> bool:
        return not self.is_write


class AccessCollector(NodeVisitor):
    """Collects all field accesses and corresponding offsets."""

    def visit_FieldAccess(
        self,
        node: oir.FieldAccess,
        *,
        accesses: List[Access],
        is_write: bool,
        **kwargs: Any,
    ) -> None:
        self.visit(
            node.offset, accesses=accesses, field_name=node.name, is_write=is_write, **kwargs
        )

    def visit_CartesianOffset(
        self,
        node: common.CartesianOffset,
        *,
        accesses: List[Access],
        field_name: str,
        is_write: bool,
        region: oir.HorizontalMask = None,
        **kwargs: Any,
    ) -> None:
        offsets = node.to_dict()
        accesses.append(
            Access(
                field=field_name,
                offset=(offsets["i"], offsets["j"], offsets["k"]),
                is_write=is_write,
                in_mask=kwargs.get("in_mask", False),
                region=region,
            )
        )

    def visit_AssignStmt(
        self,
        node: oir.AssignStmt,
        **kwargs: Any,
    ) -> None:
        self.visit(node.right, is_write=False, **kwargs)
        self.visit(node.left, is_write=True, **kwargs)

    def visit_MaskStmt(self, node: oir.MaskStmt, **kwargs: Any) -> None:

        self.visit(node.mask, is_write=False, **kwargs)
        regions = node.mask.iter_tree().if_isinstance(oir.HorizontalMask).to_list()
        if regions:
            for region in regions:
                self.visit(node.body, in_mask=True, region=region, **kwargs)
        else:
            self.visit(node.body, in_mask=True, region=None, **kwargs)

    def visit_While(self, node: oir.While, **kwargs: Any) -> None:
        self.visit(node.cond, is_write=False, **kwargs)
        self.visit(node.body, **kwargs)

    @dataclass
    class Result:
        _ordered_accesses: List["Access"]

        @staticmethod
        def _offset_dict(accesses: XIterable) -> Dict[str, Set[Tuple[int, int, int]]]:
            return accesses.reduceby(
                lambda acc, x: acc | {x.offset}, "field", init=set(), as_dict=True
            )

        def offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            """Get a dictonary, mapping all accessed fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses))

        def read_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            """Get a dictonary, mapping read fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

        def read_accesses(self) -> List[Access]:
            """Get the sub-list of read accesses"""
            return list(xiter(self._ordered_accesses).filter(lambda x: x.is_read))

        def write_offsets(self) -> Dict[str, Set[Tuple[int, int, int]]]:
            """Get a dictonary, mapping written fields' names to sets of offset tuples."""
            return self._offset_dict(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

        def write_accesses(self) -> List[Access]:
            """Get the sub-list of write accesses"""
            return list(xiter(self._ordered_accesses).filter(lambda x: x.is_write))

        def fields(self) -> Set[str]:
            """Get a set of all accessed fields' names."""
            return {acc.field for acc in self._ordered_accesses}

        def read_fields(self) -> Set[str]:
            """Get a set of all read fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_read}

        def write_fields(self) -> Set[str]:
            """Get a set of all written fields' names."""
            return {acc.field for acc in self._ordered_accesses if acc.is_write}

        def mask_writes(self) -> Set[str]:
            """Get a set of all fields' names written in mask statements."""
            return {acc.field for acc in self._ordered_accesses if acc.is_write and acc.in_mask}

        def ordered_accesses(self) -> List[Access]:
            """Get a list of ordered accesses."""
            return self._ordered_accesses

    @classmethod
    def _compensate_regions(cls, result: "AccessCollector.Result"):

        res_accesses = []
        for acc in result._ordered_accesses:
            if acc.region is not None:
                res_offset = []
                for off, bound in zip(acc.offset, (acc.region.i, acc.region.j)):
                    bound = common.HorizontalInterval(start=bound.start, end=bound.end)
                    if off > 0:
                        if bound.end is None:
                            pass
                        elif bound.end.level == common.LevelMarker.END:
                            off = off + bound.end.offset
                        else:
                            off = 0
                    if off < 0:
                        if bound.start is None:
                            # bound.start = common.AxisBound.start()
                            pass
                        elif bound.start.level == common.LevelMarker.START:
                            off = off + bound.start.offset
                        else:
                            off = 0
                    res_offset.append(off)
                res_offset.append(acc.offset[2])
                res_accesses.append(
                    Access(
                        field=acc.field,
                        offset=(res_offset[0], res_offset[1], res_offset[2]),
                        is_write=acc.is_write,
                        in_mask=acc.in_mask,
                        region=acc.region,
                    )
                )
            else:
                res_accesses.append(acc)
        return AccessCollector.Result(res_accesses)

    @classmethod
    def apply(cls, node: TreeNode, *, compensate_regions=False, **kwargs: Any) -> "Result":
        result = cls.Result([])
        cls().visit(node, accesses=result._ordered_accesses, **kwargs)

        if compensate_regions:
            result = cls._compensate_regions(result)
            return result

        return result


def symbol_name_creator(used_names: Set[str]) -> Callable[[str], str]:
    """Create a function that generates symbol names that are not already in use.

    Args:
        used_names: Symbol names that are already in use and thus should not be generated.
                    NOTE: `used_names` will be modified to contain all generated symbols.

    Returns:
        A callable to generate new unique symbol names.
    """

    def increment_string_suffix(s: str) -> str:
        if not s[-1].isnumeric():
            return s + "0"
        return re.sub(r"[0-9]+$", lambda n: str(int(n.group()) + 1), s)

    def new_symbol_name(name: str) -> str:
        while name in used_names:
            name = increment_string_suffix(name)
        used_names.add(name)
        return name

    return new_symbol_name
