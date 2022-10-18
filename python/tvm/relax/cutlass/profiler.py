import click
import tvm

from tvm.contrib.cutlass.build import select_gemm_kernel, _get_cutlass_path
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler

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


@click.command()
@click.option("--m", default=1024, help="M dimension")
@click.option("--n", default=1024, help="N dimension")
@click.option("--k", default=1024, help="K dimension")
@click.option("--sm", default=80, help="SM version")
@click.option("--typea", default="float16", help="Type of A")
@click.option("--typeb", default="float16", help="Type of B")
@click.option("--typec", default="float16", help="Type of C")
@click.option("--layouta", default="row", help="Layout of A")
@click.option("--layoutb", default="row", help="Layout of B")
@click.option("--layoutc", default="row", help="Layout of C")
@click.option("--op_type", default="cutlass.dense", help="Epilogue pattern")
@click.option("--tmpdir", default="/tmp", help="Temp directory")
def main(m, n, k, sm, typea, typeb, typec, layouta, layoutb, layoutc, op_type, tmpdir):
    cutlass_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), tmpdir)
    name, cutlass_op_def = select_gemm_kernel(
        cutlass_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        False,
        False,
        True,
    )
    print(name)
    print(cutlass_op_def)


if __name__ == "__main__":
    main()
