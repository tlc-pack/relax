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
 * \file resize.cc
 * \brief Image resize operators.
 */

#include "resize.h"

#include <utility>

namespace tvm {
namespace relax {

/* relax.resize2d */
TVM_REGISTER_NODE_TYPE(Resize2DAttrs);

Expr resize2d(Expr data, Array<PrimExpr> size, Array<FloatImm> roi, String layout, String method,
              String coordinate_transformation_mode, String rounding_method, double cubic_alpha,
              int cubic_exclude, double extrapolation_value, DataType out_dtype) {
  if (size.size() == 1) {
    size.push_back(size[0]);
  }
  CHECK_EQ(size.size(), 2) << "In Resize2D, the input size length is expected to be 2. However, "
                              "the given size is "
                           << size;

  ObjectPtr<Resize2DAttrs> attrs = make_object<Resize2DAttrs>();
  attrs->size = std::move(size);
  attrs->roi = std::move(roi);
  attrs->layout = std::move(layout);
  attrs->method = std::move(method);
  attrs->coordinate_transformation_mode = std::move(coordinate_transformation_mode);
  attrs->rounding_method = std::move(rounding_method);
  attrs->cubic_alpha = cubic_alpha;
  attrs->cubic_exclude = cubic_exclude;
  attrs->extrapolation_value = extrapolation_value;
  attrs->out_dtype = out_dtype;

  static const Op& op = Op::Get("relax.image.resize2d");
  return Call(op, {std::move(data)}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relax.op.image.resize2d").set_body_typed(resize2d);

StructInfo InferStructInfoResize2D(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo data_sinfo = GetUnaryInputTensorStructInfo(call, ctx);

  const auto* attrs = call->attrs.as<Resize2DAttrs>();
  auto [data_layout, data2NCHW] = CheckTensorLayout(call, ctx, attrs->layout,  //
                                                    /*tgt_layout=*/"NCHW",     //
                                                    /*tensor_name=*/"data");

  DataType out_dtype = attrs->out_dtype.is_void() ? data_sinfo->dtype : attrs->out_dtype;

  Optional<ShapeExpr> data_shape =
      CheckNdimPerLayoutAndGetShape(call, ctx, data_sinfo, data_layout);
  if (!data_shape.defined()) {
    return TensorStructInfo(out_dtype, data_layout.ndim());
  }

  Array<PrimExpr> data_NCHW_shape = data2NCHW.ForwardShape(data_shape.value()->values);
  Array<PrimExpr> out_NCHW_shape(data_NCHW_shape);
  out_NCHW_shape.Set(2, attrs->size[0]);
  out_NCHW_shape.Set(3, attrs->size[1]);

  Array<PrimExpr> out_shape = data2NCHW.BackwardShape(out_NCHW_shape);
  return TensorStructInfo(ShapeExpr(out_shape), out_dtype);
}

TVM_REGISTER_OP("relax.image.resize2d")
    .set_attrs_type<Resize2DAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoResize2D);

}  // namespace relax
}  // namespace tvm
