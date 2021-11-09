import tvm
from tvm import tir, te
from tvm import relax as rx

@tvm.register_func("mybuild")
def mybuild(mod, target, target_host):
    return tvm.build(mod, target, target_host)

bb = rx.BlockBuilder()
n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
x = rx.Var("x", [n, m])

with bb.function([x], "rx_func"):
    A = rx.te_tensor(x, name="A")
    B = te.compute((n, m), lambda i, j: A[i, j] * 2, name="B")
    b = bb.emit(B)
    bb.emit_func_output(b)

func = bb.get()
tir_mod = bb._tir_mod

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
