#include <tvm/relay/op.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

using tvm::relay::Call;

Expr MakeCallDPS(Expr shape, Expr func, Expr args) {
  static const Op& op = Op::Get("call_dps");
  return Call(op, {shape, func, args}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op._make.call_dps")
.set_body_typed(MakeCallDPS);

RELAY_REGISTER_OP("call_dps")
.set_num_inputs(3)
.add_argument("shape", "ShapeTuple", "The output shape.")
.add_argument("func", "Function", "TIR function or packed function.")
.add_argument("args", "Tuple", "The input arguments.");

} // namespace relax
} // namespace tvm
