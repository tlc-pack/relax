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

import tempfile
import onnx
import tvm
from tvm import relax
import numpy as np
from onnx import helper, TensorProto
from tvm import meta_schedule as ms
import onnxruntime
import run_cutlass_tuning

SRC_FILE = "./fmha.cu"
PKG_FILE = "/tmp/packaged.so"

BATCH_SIZE = 1
SEQ_LEN = 512
NUM_HEADS = 12
HEAD_SIZE = 64


"""
Input to FusedQKVToCxt:
    qkv: [batch_size, seq_len, 3 * num_heads * head_size], "float32"
    mask: [batch_size, seq_len], "int32"
    num_heads: "int32"
    output: [batch_size, num_heads, seq_len, head_size], "float32"
"""

QKV_SHAPE = (BATCH_SIZE, SEQ_LEN, 3 * NUM_HEADS * HEAD_SIZE)
MASK_SHAPE = (BATCH_SIZE, SEQ_LEN)
OUTPUT_SHAPE = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_SIZE)

print(f"QKV: {QKV_SHAPE}")
print(f"MASK: {MASK_SHAPE}")
print(f"OUTPUT: {OUTPUT_SHAPE}")


def import_source_module(executable):
    code = open(SRC_FILE, "r").read()
    fmt = "cu"
    func_names = ["whatever.cu"]
    const_vars = []  # type: ignore
    mod = tvm.get_global_func("runtime.CSourceModuleCreate")(
        code,
        fmt,
        func_names,
        const_vars,
    )
    executable.mod.import_module(mod)


def test_construct_onnx_graph():
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

    hidden_size = 384
    batch_size = 4
    sequence_length = 4
    num_heads = 12
    head_size = 32

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    weight = np.random.normal(size=(hidden_size, 3 * hidden_size)).astype(dtype) * 0.1
    bias = np.random.randn(3 * hidden_size).astype(dtype)
    mask_index = np.full((batch_size, sequence_length), 1).astype("int32")

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
    attention_node = helper.make_node(
        "Attention",
        [
            "input_emb",
            "attention_weight",
            "attention_bias",
            "attention_mask",
        ],
        ["attention_out"],
        num_heads=num_heads,
    )

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
            attention_node,
        ],
        "simple_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape)),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, list(c_shape)),
            helper.make_tensor_value_info("input_emb", TensorProto.FLOAT, list(input_array.shape)),
            helper.make_tensor_value_info(
                "attention_weight", TensorProto.FLOAT, list(weight.shape)
            ),
            helper.make_tensor_value_info("attention_bias", TensorProto.FLOAT, list(bias.shape)),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.FLOAT, list(mask_index.shape)
            ),
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


if __name__ == "__main__":
    model = onnx.load(
        "./onnx_emails_int32_dummy_turing_vortex_fixed_v2.onnx"
    )

    shape_dict = {
        "q_title_token_ids": [1, 512],
        "q_title_token_types": [1, 512],
        "q_title_token_masks": [1, 512],
    }

    mod = relax.frontends.from_onnx(model, shape=shape_dict)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

    mod = relax.transform.LowerVtxMM()(mod)
    # print(mod["vtx_mm_0"].attrs["c_source"])
    # print(mod.script())
    print("Transformed to Cutlass")

    target = tvm.target.Target("cuda -arch=sm_80 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32 -registers_per_block=65536")
    with target:
        database = ms.database.JSONDatabase("./workload.json", "./records.json")
        print("Database Loaded")
        with tempfile.TemporaryDirectory() as work_dir:
            #db = ms.relax_integration.tune_relax(
            #    mod=mod,
            #    target=target,
            #    params=None,
            #    num_trials_per_iter=4,
            #    max_trials_per_task=4,
            #    max_trials_global=160,
            #    work_dir=work_dir,
            #    database=database,
            #)
            relax_ex = ms.relax_integration.compile_relax(
                database,
                mod=mod,
                target=target,
                params=None,
            )
            print("Compiled")
        import_source_module(relax_ex)
        relax_ex.mod.export_library(
            PKG_FILE,
            cc="nvcc",
        )
        executable = tvm.runtime.load_module(PKG_FILE)
        vm = relax.VirtualMachine(executable, tvm.cuda())
    print("VM Created")
    input0 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())
    input1 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())
    input2 = tvm.nd.array(np.random.rand(1, 512).astype("int32"), tvm.cuda())
    res = vm["main"](input0, input1, input2)
    print("GPU: ", res[0], res[1], res[2])