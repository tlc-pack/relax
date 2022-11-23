# pylint: disable=missing-docstring
# pylint: disable=consider-using-f-string

# Supported op_type:
# "cutlass.dense": (EpilogueFunctor.LinearCombination, False),
# "cutlass.dense_bias": (EpilogueFunctor.LinearCombinationBias, True),
# "cutlass.dense_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
# "cutlass.dense_bias_gelu_fp16": (EpilogueFunctor.LinearCombinationGelu, False),
# "cutlass.dense_bias_gelu_fp32": (EpilogueFunctor.LinearCombinationGelu, False),
# "cutlass.batch_matmul": (EpilogueFunctor.LinearCombination, False),
# "cutlass.conv2d_bias_hardswish": (EpilogueFunctor.LinearCombinationHardSwish, False),
# "cutlass.conv2d_bias_silu": (EpilogueFunctor.LinearCombinationSilu, False),
# "cutlass.conv2d_bias_sigmoid": (EpilogueFunctor.LinearCombinationSigmoid, False),
# "cutlass.conv2d_bias_relu": (EpilogueFunctor.LinearCombinationRelu, True),
# "cutlass.conv2d_bias": (EpilogueFunctor.LinearCombinationBias, True),
# "cutlass.conv2d": (EpilogueFunctor.LinearCombination, False),
# "cutlass.conv2d_transpose": (EpilogueFunctor.LinearCombination, False),
# "cutlass.conv2d_backward_weight": (EpilogueFunctor.LinearCombination, False),

import os

from tvm._ffi import register_func
from tvm.contrib.cutlass.build import _get_cutlass_path, select_gemm_kernel
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler
from tvm.contrib.cutlass.library import DataTypeTag

SM = 80  # T4 GPU is sm_75, but for some reason it's not supported
TMP_DIR = "/tmp/"


def get_template():
    vtx_home = os.path.realpath(os.path.join(os.path.dirname(__file__), "./"))
    template_path = os.path.join(vtx_home, "gemm.cu")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


@register_func("tvm.relax.vtx.cutlass_gemm")
def _cutlass_gemm(
    func_name,
    m,
    n,
    k,
    type_a,  # "float32", "float16", "int8", "uint8",
    type_b,
    type_c,
    layout_a,  # "row", "col"
    layout_b,
    layout_c,
    op_type,
):
    cutlass_profiler = CutlassGemmProfiler(SM, _get_cutlass_path(), TMP_DIR)
    operator_name, operator_def, op = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        m,
        k,
        n,
        type_c,
        type_a,
        type_b,
        layout_c,
        layout_a,
        layout_b,
        use_3xtf32=False,
        batched=False,
        find_first_valid=False,
        use_multiprocessing=True,
    )
    layout = "const int layout_a = {}, layout_b = {}, layout_c = {};\n".format(
        1 if layout_a == "col" else 0,
        1 if layout_b == "col" else 0,
        1 if layout_c == "col" else 0,
    )
    leading_dim = op.leading_dim()
    dtype_def = "using DTypeA = {}; using DTypeB = {}; using DTypeC = {};\n".format(
        DataTypeTag[op.A.element],
        DataTypeTag[op.B.element],
        DataTypeTag[op.C.element],
    )
    source_code = (
        get_template()
        .replace("{{Layout}}", layout)
        .replace("{{LeadingDim}}", leading_dim)
        .replace("{{DTypeDef}}", dtype_def)
        .replace("{{OperatorDef}}", operator_def)
        .replace("{{OperatorName}}", operator_name)
        .replace("{{FUNC_NAME}}", func_name)
    )
    return source_code


def cutlass_gemm(
    func_name,
    m,
    n,
    k,
    type_a,  # "float32", "float16", "int8", "uint8",
    type_b,
    type_c,
    layout_a,  # "row", "col"
    layout_b,
    layout_c,
    op_type,
):
    return _cutlass_gemm(
        func_name,
        m,
        n,
        k,
        type_a,
        type_b,
        type_c,
        layout_a,
        layout_b,
        layout_c,
        op_type,
    )


if __name__ == "__main__":
    print(
        cutlass_gemm(
            func_name="my_func",
            m=4096,
            n=4096,
            k=4096,
            type_a="float32",
            type_b="float32",
            type_c="float32",
            layout_a="row",
            layout_b="row",
            layout_c="row",
            op_type="cutlass.dense",
        )
    )
