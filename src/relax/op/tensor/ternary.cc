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

/*!
 * \file ternary.cc
 * \brief ternary operators.
 */

#include "ternary.h"

namespace tvm {
namespace relax {

RELAY_REGISTER_OP("relax.ewise_fma")
    .set_num_inputs(3)
    .add_argument("e1", "Expr", "The input expression")
    .add_argument("e2", "Expr", "The input expression")
    .add_argument("e3", "Expr", "The input expression")
    .set_attr<FInferShape>("FInferShape", InferShapeEwiseFMA)
    .set_attr<FInferType>("FInferType", InferTypeEwiseFMA);

Expr MakeEwiseFma(Expr expr1, Expr expr2, Expr expr3) {
  static const Op& op = Op::Get("relax.ewise_fma");
  return Call(op, {expr1, expr2, expr3}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.ewise_fma").set_body_typed(MakeEwiseFma);

}  // namespace relax
}  // namespace tvm
