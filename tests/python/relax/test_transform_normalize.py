# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest

import tvm
import tvm.testing
from tvm import relax
from tvm import tir
from tvm.ir.base import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


def test_normalize_function():
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    type_anno = relax.DynTensorType(ndim=2, dtype="float16")
    x = relax.Var("x", [m, n], type_anno)

    # Note: the parser automatically normalize the IR written in TVMScript,
    # so we manually construct the function here.
    mul_add = relax.Function(
        [x],
        relax.op.multiply(relax.op.add(x, x), relax.op.add(x, x)),
        ret_type=type_anno,
        ret_shape=relax.RuntimeDepShape(),
    )

    # Note: from_expr api names private function (function without global_symbol) as "main"
    before_mod = tvm.IRModule.from_expr(mul_add)

    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(x: R.Tensor(("m", "n"), "float16")) -> R.Tensor(dtype="float16", ndim=2):
        gv = R.add(x, x)
        gv1 = R.add(x, x)
        return R.multiply(gv, gv1)

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_if():
    cond = relax.Var("cond", [], type_annotation=relax.DynTensorType(0, "bool"))
    x = relax.Var("x", [tir.IntImm("int64", 1)], type_annotation=relax.DynTensorType(1, "float32"))
    # TODO(relax-team): add type and shape inference for IfNode
    y = relax.Var("y")

    # Note: the parser automatically normalize the IR written in TVMScript,
    # so we manually construct the function and If here.
    f = relax.Function(
        [cond, x],
        relax.SeqExpr(
            [
                relax.BindingBlock(
                    [
                        relax.VarBinding(
                            y,
                            relax.If(
                                cond,
                                relax.op.multiply(relax.op.add(x, x), relax.op.add(x, x)),
                                relax.op.add(relax.op.multiply(x, x), relax.op.multiply(x, x)),
                            ),
                        )
                    ]
                )
            ],
            y,
        ),
        ret_type=relax.DynTensorType(1, "float32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(
        cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
    ) -> R.Tensor(dtype="float32", ndim=1):
        if cond:
            gv = R.add(x, x)
            gv1 = R.add(x, x)
            y = R.multiply(gv, gv1)
        else:
            gv = R.multiply(x, x)
            gv1 = R.multiply(x, x)
            y = R.add(gv, gv1)
        return y

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_no_op():
    # the normalize pass should be no-op for IR in ANF
    @tvm.script.ir_module
    class ANFMod1:
        @R.function
        def f(x: R.Tensor(dtype="float32")):
            gv = R.add(x, x)
            gv1 = R.add(gv, gv)
            gv2 = R.add(gv, gv1)
            return (gv, gv2)

    before_mod = ANFMod1
    after_mod = relax.transform.Normalize()(before_mod)
    assert_structural_equal(before_mod, after_mod, map_free_vars=True)

    @tvm.script.ir_module
    class ANFMod2:
        @R.function
        def foo(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.var("int64"), T.var("int64")
            with R.dataflow():
                lv0 = R.call_tir("test.op.identity", (x,), (m, n), dtype="float32")
                gv0 = R.call_tir("test.op.identity", (lv0,), (m, n), dtype="float32")
                R.output(gv0)
            return gv0

    mod = ANFMod2
    mod_post = relax.transform.Normalize()(mod)

    assert_structural_equal(mod, mod_post)


def test_normalize_seq_body():
    # a seq expression with a non-leaf body should bind the body to a var as well
    x = relax.Var("x", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    seq = relax.SeqExpr([], relax.op.add(x, y))
    f = relax.Function(
        [x, y],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(
        x: R.Tensor((), dtype="int32"), y: R.Tensor((), dtype="int32")
    ) -> R.Tensor(ndim=0, dtype="int32"):
        # normalization inserts a binding like this
        z = R.add(x, y)
        return z

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_func_body():
    # a function with a body that is not a seq expr should have it wrapped in a seq expr
    x = relax.Var("x", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    f = relax.Function(
        [x, y],
        relax.op.add(x, y),
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(
        x: R.Tensor((), dtype="int32"), y: R.Tensor((), dtype="int32")
    ) -> R.Tensor(ndim=0, dtype="int32"):
        # result will be a seq expr where the body is a var
        z = R.add(x, y)
        return z

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_if_branches():
    # an if node's branches must be seq exprs
    x = relax.Var("x", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], type_annotation=relax.DynTensorType(ndim=0, dtype="int32"))
    # TODO(@relax-team): z has a shape of () and type of DynTensorType(ndim=0),
    # but normalization fails to infer these even though it should
    z = relax.Var("z")
    cond = relax.Var("cond", [], type_annotation=relax.DynTensorType(ndim=0, dtype="bool"))
    plus = relax.op.add(x, y)
    mult = relax.op.multiply(x, y)
    if_node = relax.If(cond, plus, mult)
    seq = relax.SeqExpr([relax.BindingBlock([relax.VarBinding(z, if_node)])], z)
    f = relax.Function(
        [cond, x, y],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(
        cond: R.Tensor((), dtype="bool"),
        x: R.Tensor((), dtype="int32"),
        y: R.Tensor((), dtype="int32"),
    ) -> R.Tensor(ndim=0, dtype="int32"):
        # the bodies of the branches will be seq exprs with a binding
        if cond:
            w = R.add(x, y)
            z = w
        else:
            w = R.multiply(x, y)
            z = w
        return z

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_if_condition():
    cond = relax.Var("cond", [], type_annotation=relax.DynTensorType(0, "bool"))
    x = relax.Var("x", [tir.IntImm("int64", 1)], type_annotation=relax.DynTensorType(1, "float32"))
    # TODO(relax-team): add type and shape inference for IfNode
    y = relax.Var("y")

    # The condition is wrapped in a tuple and then indexed
    f = relax.Function(
        [cond, x],
        relax.SeqExpr(
            [
                relax.BindingBlock(
                    [
                        relax.VarBinding(
                            y,
                            relax.If(
                                relax.TupleGetItem(relax.Tuple([cond]), 0),
                                relax.op.add(x, x),
                                relax.op.multiply(x, x),
                            ),
                        )
                    ]
                )
            ],
            y,
        ),
        ret_type=relax.DynTensorType(1, "float32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    @R.function
    def expected(
        cond: R.Tensor((), "bool"), x: R.Tensor((1,), "float32")
    ) -> R.Tensor(dtype="float32", ndim=1):
        c = R.TupleGetItem(R.Tuple(cond), 0)
        if c:
            gv = R.add(x, x)
            y = gv
        else:
            gv = R.multiply(x, x)
            y = gv
        return y

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_tuple_get_item():
    x = relax.Var("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    f = relax.Function(
        [x],
        relax.TupleGetItem(
            relax.TupleGetItem(
                relax.Tuple([relax.Tuple([x])]),
                0,
            ),
            0,
        ),
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    before_mod = tvm.IRModule.from_expr(f)
    after_mod = relax.transform.Normalize()(before_mod)

    # TODO: Revisit once we canonicalize SeqExprs (part of normalization?)
    # Not using the parser this time because writing it out correctly results in
    # *one* binding block, whereas the normalized version has *two*
    idx_var = relax.Var(
        "idx_var",
        shape_annotation=relax.Tuple([relax.ShapeExpr([])]),
        type_annotation=relax.TupleType([relax.DynTensorType(ndim=0, dtype="int32")]),
    )
    ret_var = relax.Var("ret", [], relax.DynTensorType(ndim=0, dtype="int32"))
    expected_f = relax.Function(
        [x],
        relax.SeqExpr(
            [
                relax.BindingBlock(
                    [
                        relax.VarBinding(
                            idx_var, relax.TupleGetItem(relax.Tuple([relax.Tuple([x])]), 0)
                        )
                    ]
                ),
                relax.BindingBlock([relax.VarBinding(ret_var, relax.TupleGetItem(idx_var, 0))]),
            ],
            ret_var,
        ),
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )
    expected_mod = tvm.IRModule.from_expr(expected_f)
    # apply normalization to fill in type and shape annotations (tedious otherwise)
    final_mod = relax.transform.Normalize()(expected_mod)

    assert_structural_equal(after_mod, final_mod)


def test_normalize_combine_nearby_blocks():
    x = relax.Var("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    v0 = relax.Var("v0", [], relax.DynTensorType(ndim=0, dtype="int32"))
    v1 = relax.Var("v1", [], relax.DynTensorType(ndim=0, dtype="int32"))
    v2 = relax.Var("v2", [], relax.DynTensorType(ndim=0, dtype="int32"))
    v3 = relax.Var("v3", [], relax.DynTensorType(ndim=0, dtype="int32"))
    f = relax.Function(
        [x],
        relax.SeqExpr(
            [
                relax.DataflowBlock([relax.VarBinding(v0, x)]),
                relax.DataflowBlock([relax.VarBinding(v1, v0)]),
                relax.BindingBlock([relax.VarBinding(v2, v1)]),
                relax.BindingBlock([relax.VarBinding(v3, v2)]),
            ],
            v3,
        ),
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )

    after_mod = relax.transform.Normalize()(tvm.IRModule.from_expr(f))

    @R.function
    def expected(x: R.Tensor((), "int32")):
        with R.dataflow():
            v0 = x
            v1 = v0
            R.output(v0, v1)
        v2 = v1
        v3 = v2
        return v3

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_nested_seq():
    x = relax.Var("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], relax.DynTensorType(ndim=0, dtype="int32"))
    z = relax.Var("z", [], relax.DynTensorType(ndim=0, dtype="int32"))
    seq = relax.SeqExpr(
        [
            relax.BindingBlock(
                [
                    relax.VarBinding(x, relax.const(1)),
                    relax.VarBinding(
                        y,
                        relax.SeqExpr(
                            [relax.BindingBlock([relax.VarBinding(z, relax.const(2))])],
                            z,
                        ),
                    ),
                ]
            )
        ],
        y,
    )

    f = relax.Function(
        [],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )
    after_mod = relax.transform.Normalize()(tvm.IRModule.from_expr(f))

    @R.function
    def expected():
        x = relax.const(1)
        z = relax.const(2)
        y = z
        return y

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_nested_seq_dataflow():
    x = relax.Var("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], relax.DynTensorType(ndim=0, dtype="int32"))
    z = relax.Var("z", [], relax.DynTensorType(ndim=0, dtype="int32"))
    q = relax.Var("u", [], relax.DynTensorType(ndim=0, dtype="int32"))
    w = relax.DataflowVar("w", [], relax.DynTensorType(ndim=0, dtype="int32"))
    u = relax.Var("u", [], relax.DynTensorType(ndim=0, dtype="int32"))
    seq = relax.SeqExpr(
        [
            relax.BindingBlock(
                [
                    relax.VarBinding(x, relax.const(1)),
                    relax.VarBinding(
                        y,
                        relax.SeqExpr(
                            [
                                relax.BindingBlock([relax.VarBinding(q, relax.const(2))]),
                                relax.DataflowBlock(
                                    [
                                        relax.VarBinding(w, q),
                                        relax.VarBinding(u, w),
                                    ]
                                ),
                                relax.BindingBlock([relax.VarBinding(z, u)]),
                            ],
                            z,
                        ),
                    ),
                ]
            )
        ],
        y,
    )

    f = relax.Function(
        [],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )
    after_mod = relax.transform.Normalize()(tvm.IRModule.from_expr(f))

    @R.function
    def expected():
        x = relax.const(1)
        q = relax.const(2)
        with R.dataflow():
            w = q
            u = w
            R.output(u)
        z = u
        y = z
        return y

    assert_structural_equal(after_mod["main"], expected)


def test_normalize_deeply_nested_seq():
    x = relax.Var("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], relax.DynTensorType(ndim=0, dtype="int32"))
    z = relax.Var("z", [], relax.DynTensorType(ndim=0, dtype="int32"))
    u = relax.Var("u", [], relax.DynTensorType(ndim=0, dtype="int32"))
    v = relax.Var("v", [], relax.DynTensorType(ndim=0, dtype="int32"))
    w = relax.Var("w", [], relax.DynTensorType(ndim=0, dtype="int32"))
    seq = relax.SeqExpr(
        [
            relax.BindingBlock(
                [
                    relax.VarBinding(x, relax.const(1)),
                    relax.VarBinding(
                        y,
                        relax.SeqExpr(
                            [
                                relax.BindingBlock(
                                    [
                                        relax.VarBinding(
                                            z,
                                            relax.SeqExpr(
                                                [
                                                    relax.BindingBlock(
                                                        [
                                                            relax.VarBinding(u, relax.const(2)),
                                                            relax.MatchShape(u, [], None),
                                                            relax.VarBinding(v, u),
                                                            relax.MatchShape(v, [], w),
                                                        ]
                                                    )
                                                ],
                                                w,
                                            ),
                                        )
                                    ]
                                )
                            ],
                            z,
                        ),
                    ),
                ]
            )
        ],
        y,
    )

    f = relax.Function(
        [],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )
    after_mod = relax.transform.Normalize()(tvm.IRModule.from_expr(f))

    @R.function
    def expected():
        x = relax.const(1)
        u = relax.const(2)
        R.match_shape(u, ())
        v = u
        w = R.match_shape(v, ())
        z = w
        y = z
        return y

    assert_structural_equal(after_mod["main"], expected)


@pytest.mark.xfail()
def test_nesting_non_dataflow_in_dataflow_error():
    x = relax.DataflowVar("x", [], relax.DynTensorType(ndim=0, dtype="int32"))
    y = relax.Var("y", [], relax.DynTensorType(ndim=0, dtype="int32"))
    z = relax.Var("z", [], relax.DynTensorType(ndim=0, dtype="int32"))
    seq = relax.SeqExpr(
        [
            relax.DataflowBlock(
                [
                    relax.VarBinding(x, relax.const(1)),
                    relax.VarBinding(
                        y,
                        relax.SeqExpr(
                            [relax.BindingBlock([relax.VarBinding(z, relax.const(2))])],
                            z,
                        ),
                    ),
                ]
            )
        ],
        y,
    )
    f = relax.Function(
        [],
        seq,
        ret_type=relax.DynTensorType(ndim=0, dtype="int32"),
        ret_shape=relax.RuntimeDepShape(),
    )
    relax.transform.Normalize()(tvm.IRModule.from_expr(f))
    # should fail due to a normal binding block being inside a dataflowblock


if __name__ == "__main__":
    tvm.testing.main()
