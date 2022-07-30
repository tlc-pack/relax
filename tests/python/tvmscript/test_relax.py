from tvm.script.parser import tir as T, relax as R
import tvm.testing
from tvm import IRModule


def _check(mod: IRModule):
    print(mod.script())


def test_simple_module():
    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def tir_func(x: T.Buffer((128, 128), "float32"), y: T.Buffer((128, 128), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv0 = R.call_tir(tir_func, x, (128, 128), dtype="float32")
            return gv0

    _check(TestModule)


def test_simple_func():
    @R.function
    def test_func(x: R.Tensor((128, 128), "float32")) -> R.Tensor(None, "float32", ndim=2):
        gv0 = R.call_tir("extern_func", x, (128, 128), dtype="float32")
        return gv0

    _check(test_func)


def test_symbolic_shape():
    @R.function
    def test_func(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        m = T.var("int32", "m")
        n = T.var("int32", "n")
        gv0 = R.call_tir("extern_func", x, (m, n), dtype="float32")
        return gv0

    _check(test_func)


def test_directly_return():
    @R.function
    def test_func(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(None, "float32", ndim=2):
        return x

    _check(test_func)


def test_call_packed():
    @R.function
    def test_func(x: R.Tensor((3, 4), "float32")) -> R.Tensor(None, "float32", ndim=2):
        z = R.call_packed("vm.builtin.copy", x, type_args=R.Tensor(None, "float32", ndim=2))
        return z

    _check(test_func)


def test_relax_op():
    @R.function
    def test_func(
        x: R.Tensor((4, 4), "float32"), w: R.Tensor((4, 4), "float32")
    ) -> R.Tensor(None, "float32", ndim=2):
        y = R.add(x, w)
        z = R.multiply(x, y)
        return z

    _check(test_func)


if __name__ == "__main__":
    test_call_packed()
    tvm.testing.main()
