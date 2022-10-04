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

from __future__ import annotations  # must import to defer parsing of annotations
import pytest
import numpy as np
import tvm
from tvm import relay, relax
from tvm.contrib import graph_executor
from tvm.relax.testing import transform
from tvm.script import relax as R, tir as T
import tvm.testing

target_str = "llvm --num-cores=16"
target = tvm.target.Target(target_str)
dev = tvm.device(target_str, 0)


def test_dense():
    # Relay output
    dtype = "float32"
    X_shape = (4, 8)
    Y_shape = (4, 8)

    data_x = tvm.nd.array(np.random.rand(*X_shape).astype(np.float32), dev)
    data_y = tvm.nd.array(np.random.rand(*Y_shape).astype(np.float32), dev)

    X = relay.var("x", shape=X_shape, dtype=dtype)
    Y = relay.var("y", shape=Y_shape, dtype=dtype)
    Z = relay.nn.dense(X, Y)
    f = relay.Function([X, Y], Z)
    with target:
        lib = relay.build(f)
        m = graph_executor.GraphModule(lib["default"](dev))

        # Setup execution
        m.set_input("x", data_x)
        m.set_input("y", data_y)

        m.run()
        # get output
        expected = m.get_output(0)

    ret_sinfo = R.Tensor(dtype, ndim=2)
    X = relax.Var("x", R.Tensor(X_shape, dtype))
    Y = relax.Var("y", R.Tensor(Y_shape, dtype))
    Z = relax.nn.dense(X, Y)
    f = relax.Function([X, Y], Z, ret_sinfo)
    f = f.with_attr("global_symbol", "default")

    mod = tvm.IRModule.from_expr(f)
    with tvm.transform.PassContext(opt_level=3):
        mod = relax.transform.Normalize()(mod)
        mod = transform.LowerWithRelayOpStrategyPass(target)(mod)
        ex = relax.vm.build(mod, target)
        vm = relax.VirtualMachine(ex, dev)
        out = vm["default"](data_x, data_y)

    tvm.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_dense()
    # pytest.main([__file__])
