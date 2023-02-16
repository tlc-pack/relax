import tvm
from tvm import relax
from tvm.script import relax as R, tir as T

# Case 1: Stride with stride of non-ones. This works fine.
@tvm.script.ir_module
class SliceStrideNonOne:
    @R.function
    def main(
        x: R.Tensor((8, 9, 10, 10), dtype="float32")
    ) -> R.Tensor((4, 9, 10, 3), dtype="float32"):
        gv = R.call_tir(strided_slice, (x,), R.Tensor((4, 9, 10, 3), dtype="float32"))
        return gv

    @T.prim_func
    def strided_slice(
        rxplaceholder: T.Buffer[(T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"],
        T_strided_slice_with_axes: T.Buffer[
            (T.int64(4), T.int64(9), T.int64(10), T.int64(3)), "float32"
        ],
    ):
        T.func_attr({"tir.noalias": True})
        for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(9), T.int64(10), T.int64(3)):
            with T.block("T_strided_slice_with_axes"):
                ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(
                    rxplaceholder[
                        ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)
                    ]
                )
                T.writes(T_strided_slice_with_axes[ax0, ax1, ax2, ax3])
                T_strided_slice_with_axes[ax0, ax1, ax2, ax3] = rxplaceholder[
                    ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)
                ]


before = SliceStrideNonOne
after = relax.transform.RewriteDataflowReshape()(before)
# passes equality test
tvm.ir.assert_structural_equal(before, after)

# Case 2: Stride with stride of ones. This throws error.
@tvm.script.ir_module
class SliceStrideOne:
    @R.function
    def main(x: R.Tensor((20, 10, 5), dtype="float32")) -> R.Tensor((3, 10, 5), dtype="float32"):
        with R.dataflow():
            lv = R.call_tir(strided_slice, (x,), out_sinfo=R.Tensor((3, 10, 5), dtype="float32"))
            gv: R.Tensor((3, 10, 5), dtype="float32") = lv
            R.output(gv)
        return gv

    @T.prim_func
    def strided_slice(
        rxplaceholder: T.Buffer((T.int64(20), T.int64(10), T.int64(5)), "float32"),
        T_strided_slice_with_axes: T.Buffer((T.int64(3), T.int64(10), T.int64(5)), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(3), T.int64(10), T.int64(5)):
            with T.block("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0, v_ax1, v_ax2]


before = SliceStrideOne
after = relax.transform.RewriteDataflowReshape()(before)
after.show()
