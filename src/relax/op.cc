/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
#include <tvm/relay/op.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {

Expr MakeCallDPS(ShapeExpr shape, BaseFunc func, Tuple args) {
  static const Op& op = Op::Get("call_dps");
  return Call(op, {shape, func, args}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op._make.call_dps")
.set_body_typed(MakeCallDPS);

RELAY_REGISTER_OP("call_dps")
.set_num_inputs(3)
.add_argument("shape", "ShapeExpr", "The output shape.")
.add_argument("func", "BaseFunc", "TIR function or packed function.")
.add_argument("args", "Tuple", "The input arguments.");

} // namespace relax
} // namespace tvm
