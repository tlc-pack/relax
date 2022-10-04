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

#include "pooling.h"

#include "../tensor/unary.h"
namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(MaxPool2dAttrs);
/*
RELAY_REGISTER_OP("relax.nn.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2dAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeMaxPool2d)
    .set_attr<FInferType>("FInferType", InferTypeUnaryBroadcast);

Expr MakeMaxPool2d(Expr data, Array<PrimExpr> kernel_size, Array<PrimExpr> stride,
                   Array<PrimExpr> padding, Array<PrimExpr> dilation) {
  auto attrs = make_object<MaxPool2dAttrs>();
  attrs->kernel_size = kernel_size;
  attrs->stride = stride;
  attrs->padding = padding;
  attrs->dilation = dilation;
  static const Op& op = Op::Get("relax.nn.max_pool2d");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(MakeMaxPool2d);
*/
}  // namespace relax
}  // namespace tvm