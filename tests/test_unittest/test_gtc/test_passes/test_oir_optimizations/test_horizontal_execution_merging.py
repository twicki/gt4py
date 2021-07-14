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

from gtc import common, oir
from gtc.passes.oir_dace_optimizations.horizontal_execution_merging import (
    graph_merge_horizontal_executions,
)
from gtc.passes.oir_optimizations.horizontal_execution_merging import OnTheFlyMerging

from ...oir_utils import (
    AssignStmtFactory,
    FieldAccessFactory,
    HorizontalExecutionFactory,
    NativeFuncCallFactory,
    StencilFactory,
    TemporaryFactory,
    VerticalLoopFactory,
    VerticalLoopSectionFactory,
)


def test_zero_extent_merging():

    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="foo", right__name="bar")]
                ),
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="baz", right__name="bar")]
                ),
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="foo", right__name="foo")]
                ),
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="foo", right__name="baz")]
                ),
            ]
        )
    )
    transformed = graph_merge_horizontal_executions(testee)
    testee_hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 1

    # assert same statements are present, uniquely
    assert (
        len(set(transformed_hexecs[0].body.index(stmt) for he in testee_hexecs for stmt in he.body))
        == 4
    )

    # assert the merging didn't reorder the statements in an invalid way
    assert transformed_hexecs[0].body[-1] == AssignStmtFactory(left__name="foo", right__name="baz")
    assert transformed_hexecs[0].body.index(testee_hexecs[0].body[0]) < transformed_hexecs[
        0
    ].body.index(testee_hexecs[2].body[0])


def test_mixed_merging():
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
                HorizontalExecutionFactory(
                    body=[
                        AssignStmtFactory(left__name="bar", right__name="foo", right__offset__i=1)
                    ]
                ),
                HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="bar")]),
            ]
        )
    )
    transformed = graph_merge_horizontal_executions(testee)
    testee_hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 2
    assert transformed_hexecs[0].body == testee_hexecs[0].body
    assert transformed_hexecs[1].body == sum((he.body for he in testee_hexecs[1:]), [])


def test_write_after_read_with_offset():
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(right__name="foo", right__offset__i=1)]
                ),
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="foo")]),
            ]
        )
    )
    transformed = graph_merge_horizontal_executions(testee)
    assert transformed == testee


def test_nonzero_extent_merging():
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="foo")]),
                HorizontalExecutionFactory(
                    body=[AssignStmtFactory(right__name="foo", right__offset__j=1)]
                ),
            ]
        )
    )
    transformed = graph_merge_horizontal_executions(testee)
    testee_hexecs = testee.vertical_loops[0].sections[0].horizontal_executions
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 1

    # based on dependency, no order is guaranteed, but the same statements must be there
    assert transformed_hexecs[0].body == sum(
        (he.body for he in testee_hexecs), []
    ) or transformed_hexecs[0].body == sum(reversed(he.body for he in testee_hexecs), [])


def test_different_iteration_spaces_param():
    # need three HE since only a read-write dependency would not be allowed to merge anyways due
    # to the read with offset. The interesting part is to enforce that the first two are not
    # merged.
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="api1")]),
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="api2")]),
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        right=NativeFuncCallFactory(
                            func=common.NativeFunction.MIN,
                            args=[
                                FieldAccessFactory(name="api1", offset__i=1),
                                FieldAccessFactory(name="api2", offset__j=1),
                            ],
                        )
                    )
                ),
            ]
        )
    )

    transformed = graph_merge_horizontal_executions(testee)
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 3


def test_different_iteration_spaces_temporary():
    # need three HE since only a read-write dependency would not be allowed to merge anyways due
    # to the read with offset. The interesting part is to enforce that the first two are not
    # merged.
    testee = StencilFactory(
        vertical_loops__0__sections__0=VerticalLoopSectionFactory(
            horizontal_executions=[
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp1")]),
                HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp2")]),
                HorizontalExecutionFactory(
                    body__0=AssignStmtFactory(
                        right=NativeFuncCallFactory(
                            func=common.NativeFunction.MIN,
                            args=[
                                FieldAccessFactory(name="tmp1", offset__i=1),
                                FieldAccessFactory(name="tmp2", offset__j=1),
                            ],
                        )
                    )
                ),
            ]
        ),
        declarations=[TemporaryFactory(name="tmp1"), TemporaryFactory(name="tmp2")],
    )

    transformed = graph_merge_horizontal_executions(testee)
    transformed_hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(transformed_hexecs) == 2


def test_on_the_fly_merging_basic():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(body=[AssignStmtFactory(right__name="tmp")]),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].declarations) == 1
    assert isinstance(hexecs[0].declarations[0], oir.LocalScalar)
    assert not transformed.declarations


def test_on_the_fly_merging_with_offsets():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 1
    assert len(hexecs[0].declarations) == 2
    assert all(isinstance(d, oir.LocalScalar) for d in hexecs[0].declarations)
    assert not transformed.declarations
    assert transformed.iter_tree().if_isinstance(oir.FieldAccess).filter(
        lambda x: x.name == "foo"
    ).getattr("offset").map(lambda o: (o.i, o.j, o.k)).to_set() == {(1, 0, 0), (0, 1, 0)}


def test_on_the_fly_merging_with_expensive_function():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(
                        left__name="tmp",
                        right=NativeFuncCallFactory(func=common.NativeFunction.SIN),
                    )
                ]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging(allow_expensive_function_duplication=False).visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_body_size_limit():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="tmp", right__name="foo")]
            ),
            HorizontalExecutionFactory(
                body=[
                    AssignStmtFactory(right__name="tmp", right__offset__i=1),
                    AssignStmtFactory(right__name="tmp", right__offset__j=1),
                ]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging(max_horizontal_execution_body_size=0).visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_api_field():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(
                body__0=AssignStmtFactory(left__name="mid", right__name="inp")
            ),
            HorizontalExecutionFactory(
                body__0=AssignStmtFactory(left__name="outp", right__name="mid")
            ),
        ]
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_field_read_later():
    testee = StencilFactory(
        vertical_loops=[
            VerticalLoopFactory(
                sections__0__horizontal_executions=[
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(left__name="mid", right__name="inp")]
                    ),
                    HorizontalExecutionFactory(
                        body=[AssignStmtFactory(left__name="outp1", right__name="mid")]
                    ),
                ]
            ),
            VerticalLoopFactory(
                sections__0__horizontal_executions__0=HorizontalExecutionFactory(
                    body=[AssignStmtFactory(left__name="outp2", right__name="mid")]
                )
            ),
        ]
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2


def test_on_the_fly_merging_repeated():
    testee = StencilFactory(
        vertical_loops__0__sections__0__horizontal_executions=[
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out1", right__name="tmp")]
            ),
            HorizontalExecutionFactory(body=[AssignStmtFactory(left__name="tmp")]),
            HorizontalExecutionFactory(
                body=[AssignStmtFactory(left__name="out2", right__name="tmp")]
            ),
        ],
        declarations=[TemporaryFactory(name="tmp")],
    )
    transformed = OnTheFlyMerging().visit(testee)
    hexecs = transformed.vertical_loops[0].sections[0].horizontal_executions
    assert len(hexecs) == 2
