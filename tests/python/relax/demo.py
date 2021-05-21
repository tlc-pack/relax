import numpy as np
import tvm
from tvm import relax as rx

@tvm.register_func("vm.func0")
def func0():
    print("lalal0")

@tvm.register_func("vm.func1")
def func1():
    print("lalal1")

print("create builder")
ib = rx.Builder()
print(ib)

arr = tvm.nd.array(np.random.rand(32, 32))

ib.emit_call("vm.func0", args=[ib.r(0), ib.r(1)])
ib.emit_call("vm.func1", args=[ib.r(2), ib.imm(7482)], ret=ib.r(5))
ib.emit_call("vm.func1", args=[ib.r(3), arr])

ib.get_source()

executable = ib.get()
print(executable)
exit(0)

vm = rx.VirtualMachine()
vm.load(executable)
vm.execute()

