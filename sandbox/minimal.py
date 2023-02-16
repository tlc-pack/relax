import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op
from tvm.script import relax as R
import numpy as np
import tempfile
from tvm.relax.transform.tuning_api import Trace

# Target cpu
# dev = tvm.cpu()
# target_str = "llvm"

# Target gpu
dev = tvm.cuda()
target_str = "nvidia/geforce-rtx-3070"
target = tvm.target.Target(target_str)

m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
x = relax.Var("x", R.Tensor((m, n), "float32"))
y = relax.Var("y", R.Tensor((m, n), "float32"))
f = relax.Function(
    [x, y],
    relax.op.add(x, y),
    ret_struct_info=R.Tensor((m, n), "float32"),
)
mod = tvm.IRModule.from_expr(f)

mod = relax.transform.Normalize()(mod)


with tempfile.TemporaryDirectory() as work_dir:
    with tvm.target.Target(target), tvm.transform.PassContext(opt_level=3, trace=Trace(mod)):
        mod = relax.transform.LegalizeOps()(mod)
        mod.show()
        tuning_pass = relax.transform.MetaScheduleTuneTIR(work_dir=work_dir, max_trials_global=4)
        mod = tuning_pass(mod)

ex = relax.vm.build(mod, target, params={})
vm = relax.VirtualMachine(ex, dev)

np0 = np.random.rand(16, 16).astype(np.float32)
np1 = np.random.rand(16, 16).astype(np.float32)
data0 = tvm.nd.array(np0, dev)
data1 = tvm.nd.array(np1, dev)
inputs = [data0, data1]
out = vm["main"](*inputs)

# print(out)
