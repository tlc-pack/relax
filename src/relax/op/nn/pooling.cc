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

#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>
#include <tvm/topi/nn/pooling.h>

#include "../tensor/unary.h"
namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(MaxPool2DAttrs);

RELAY_REGISTER_OP("relax.nn.max_pool2d");

template <typename AttrType, topi::nn::PoolType mode>
Array<te::Tensor> Pool2DCompute(const Attrs& attrs, const Array<te::Tensor>& inputs,
                                const Type& out_type) {
  static const tir::Layout kNCHW("NCHW");
  const auto* param = attrs.as<AttrType>();
  ICHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto dilation = param->dilation;
  auto padding = param->padding;
  auto ceil_mode = param->ceil_mode;
  tir::Layout layout(param->layout);
  tir::Layout out_layout(param->out_layout);

  ICHECK(tir::BijectiveLayout(layout, kNCHW).defined())
      << "max_pool2d currently only supports layouts that are convertible from NCHW";
  ICHECK_EQ(layout.IndexOf(tir::LayoutAxis::Get('h')), -1)
      << "max_pool2d does not support input split on height";
  ICHECK_EQ(layout.IndexOf(tir::LayoutAxis::Get('w')), -1)
      << "max_pool2d does not support input split on width";

  if (param->padding.size() == 1) {
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
    padding.push_back(padding[0]);
  } else if (param->padding.size() == 2) {
    padding.push_back(padding[0]);
    padding.push_back(padding[1]);
  }
  if (mode == topi::nn::kAvgPool) {
    // TODO(@sunggg): Disabled for now until implementing Avg Pool
    // bool count_include_pad = reinterpret_cast<const AvgPool2DAttrs*>(param)->count_include_pad;
    // return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, stride, dilation, padding,
    //                                          mode, ceil_mode, layout.name(), count_include_pad)};
    ICHECK(0) << "Not implemented yet";
  } else {
    return Array<te::Tensor>{topi::nn::pool2d(inputs[0], pool_size, strides, dilation, padding,
                                              mode, ceil_mode, layout.name())};
  }
}

RELAX_REGISTER_OP("relax.nn.max_pool2d")
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor")
    .set_attrs_type<MaxPool2DAttrs>()
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMaxPool2D)
    .set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);

Expr MakeMaxPool2D(Expr data, Array<PrimExpr> pool_size, Array<PrimExpr> strides,
                   Array<PrimExpr> padding, Array<PrimExpr> dilation, String layout,
                   String out_layout, bool ceil_mode) {
  auto attrs = make_object<MaxPool2DAttrs>();
  attrs->pool_size = pool_size;
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->layout = layout;
  attrs->out_layout = out_layout;
  attrs->ceil_mode = ceil_mode;
  static const Op& op = Op::Get("relax.nn.max_pool2d");
  return Call(op, {data}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("relax.op.nn.max_pool2d").set_body_typed(MakeMaxPool2D);

}  // namespace relax
}  // namespace tvm
