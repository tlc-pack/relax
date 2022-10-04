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

/*!
 * \file src/relax/op/nn/convolution.cc
 * \brief Convolution operators
 */

#include "convolution.h"

//#include <tvm/relax/expr.h>
//#include <tvm/relax/op_attr_types.h>
//#include <tvm/relax/type.h>

#include "../tensor/binary.h"

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(Conv2DAttrs);

/*
Expr MakeConv2d(Expr expr1, Expr expr2, Array<PrimExpr> kernel_size, Array<PrimExpr> stride,
                Array<PrimExpr> padding, Array<PrimExpr> dilation) {
  static const Op& op = Op::Get("relax.nn.conv2d");
  auto attrs = make_object<Conv2DAttrs>();
  attrs->kernel_size = kernel_size;
  attrs->stride = stride;
  attrs->padding = padding;
  attrs->dilation = dilation;
  return Call(op, {expr1, expr2}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.nn.conv2d").set_body_typed(MakeConv2D);

RELAX_REGISTER_OP("relax.nn.conv2d")
    .describe(R"code(2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("weight", "Tensor", "The weight tensor.")
    .set_attrs_type<Conv2DAttrs>()
    .set_attr<FInferShape>("FInferShape", InferShapeConv2d)
    .set_attr<FInferType>("FInferType", InferTypeBinaryBroadcast);
*/
}  // namespace relax
}  // namespace tvm