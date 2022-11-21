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

import onnx
import tvm
from tvm import relax
import numpy as np
from onnx import helper, TensorProto


def create_initializer_tensor(
    name: str, tensor_array: np.ndarray, data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist(),
    )

    return initializer_tensor


a_shape = [128, 128]
b_shape = [128, 256]
c_shape = [256, 256]
bias_shape = [256]

# out_shape = [a_shape[0], b_shape[1]]
out_shape = c_shape

a_array = np.random.uniform(size=a_shape).astype("float32")
b_array = np.random.uniform(size=b_shape).astype("float32")
c_array = np.random.uniform(size=c_shape).astype("float32")
bias_array = np.random.rand(256).astype(np.float32)


constant_initializer_tensor = create_initializer_tensor(
    name="constant0", tensor_array=bias_array, data_type=onnx.TensorProto.FLOAT
)

constant_initializer_tensor1 = create_initializer_tensor(
    name="constant1", tensor_array=np.array(0, dtype="int64"), data_type=onnx.TensorProto.FLOAT
)

mul_node = helper.make_node("MatMul", ["a", "b"], ["out"])
mul_node2 = helper.make_node("MatMul", ["out", "c"], ["out2"])
relu_node = helper.make_node("Relu", ["out2"], ["relu_out"])
tanh_node = helper.make_node("Tanh", ["out2"], ["tanh_out"])
sigmoid_node = helper.make_node("Sigmoid", ["out2"], ["sigmoid_out"])
biasgelu_node = helper.make_node("BiasGelu", ["out2", "constant0"], ["biasgelu_out"])
gather_node = helper.make_node("Gather", ["out2", "constant1"], ["gather_out"])
concat_node = helper.make_node("Concat", ["out2", "out"], ["concat_out"])


graph = helper.make_graph(
    [
        mul_node,
        mul_node2,
        relu_node,
        tanh_node,
        sigmoid_node,
        biasgelu_node,
        gather_node,
        concat_node,
    ],
    "simple_test",
    inputs=[
        helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
        helper.make_tensor_value_info("c", TensorProto.FLOAT, list(c_shape)),
    ],
    outputs=[
        helper.make_tensor_value_info("relu_out", TensorProto.FLOAT, list(b_shape)),
        helper.make_tensor_value_info("tanh_out", TensorProto.FLOAT, list(b_shape)),
        helper.make_tensor_value_info("biasgelu_out", TensorProto.FLOAT, list(b_shape)),
    ],
    initializer=[
        constant_initializer_tensor,
        constant_initializer_tensor1,
    ],
)

model = helper.make_model(graph, producer_name="simple_test")
mod = relax.frontends.from_onnx(model)
mod.show()
