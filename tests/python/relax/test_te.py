import tvm
from tvm import tir, te
from tvm import relax as rx

bb = rx.BlockBuilder()
n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
type_anno = rx.DynTensorType(2, "float32")
x = rx.Var("x", [n, m], type_anno)
y = rx.Var("y", [n, m], type_anno)


def te_func(A, B, msg):
    print(msg)
    return te.compute((128, 128), lambda i, j: A[i, j] + B[i, j])

with bb.function([x, y], "rx_func"):
    z = bb.emit_te(te_func, x, y, msg="hello")
    bb.emit_func_output(z)

func = bb.get()
tir_mod = bb.get_tir_mod()

module = tvm.IRModule()
gvar = tvm.relay.GlobalVar("rx_func")
module[gvar] = func
for var in tir_mod:
    module[var] = tir_mod[var]

print(module.astext())

# target = tvm.target.Target("llvm")
# target_host = tvm.target.Target("llvm")
# ex, lib = rx.vm.build(module, target, target_host)
# vm = rx.VirtualMachine(ex, tvm.cpu(), mod=lib)
