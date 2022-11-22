import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T

SRC_FILE = "./apps/vtx/fmha.cu"
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


@tvm.script.ir_module
class TestModule:
    # Input IRModule.
    @R.function
    def main(
        qkv: R.Tensor((BATCH_SIZE, SEQ_LEN, 3 * NUM_HEADS * HEAD_SIZE), "float32"),
        mask: R.Tensor((BATCH_SIZE, SEQ_LEN), "int32"),
    ):
        output = relax.call_tir(
            "FusedQKVToCxt",
            (qkv, mask),
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_SIZE),
            "float32",
        )
        return output


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


def main():
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    with target:
        executable = relax.vm.build(TestModule, target=target)
        import_source_module(executable)
        executable.mod.export_library(
            PKG_FILE,
            cc="nvcc",
        )
    executable = tvm.runtime.load_module(PKG_FILE)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    qkv = tvm.nd.array(np.random.rand(*QKV_SHAPE).astype("float32"), device=tvm.cuda())
    mask = np.ones(MASK_SHAPE, "int32")
    mask[:, : SEQ_LEN // 2] = 0.0
    mask = tvm.nd.array(mask, device=tvm.cuda())
    output = vm["main"](qkv, mask)


if __name__ == "__main__":
    main()
