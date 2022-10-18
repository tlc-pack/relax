/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "nn.h"

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(DenseAttrs);

RELAX_REGISTER_OP("relax.nn.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.

- **data**: `(x1, x2, ..., xn, input_dim)`
- **weight**: `(units, input_dim)`
- **out**: `(x1, x2, ..., xn, units)`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<DenseAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "nD Tensor", "Input data.")
    .add_argument("weight", "2D Tensor", "Weight matrix.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoDense);

Expr MakeDense(Expr data, Expr weight, PrimExpr units, DataType out_dtype) {
  auto attrs = make_object<DenseAttrs>();
  attrs->units = units;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("relax.nn.dense");

  return Call(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.dense").set_body_typed(MakeDense);

TVM_REGISTER_NODE_TYPE(SoftmaxAttrs);

Expr MakeSoftmax(Expr data, int axis) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("relax.nn.softmax");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAX_REGISTER_OP("relax.nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<SoftmaxAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoUnary);

TVM_REGISTER_GLOBAL("relax.op.nn.softmax").set_body_typed(MakeSoftmax);

RELAX_REGISTER_UNARY_OP("nn.relu");

RELAX_REGISTER_OP("relax.nn.flatten")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoFlatten);

Expr MakeFlatten(Expr data) {
  static const Op& op = Op::Get("relax.nn.flatten");
  return Call(op, {data}, {}, {});
}
TVM_REGISTER_GLOBAL("relax.op.nn.flatten").set_body_typed(MakeFlatten);

}  // namespace relax
}  // namespace tvm
