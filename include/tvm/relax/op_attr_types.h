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
 * \file tvm/relax/op_attr_types.h
 * \brief Data structures that can appear in operator attributes.
 */
#ifndef TVM_RELAX_OP_ATTR_TYPES_H_
#define TVM_RELAX_OP_ATTR_TYPES_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/struct_info.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>

#include <string>

namespace tvm {
namespace relax {

/*!
 * \brief Infer output struct info given the call
 *
 * \param call The call expression to be derived.
 * \param ctx The builder context.
 */
using FInferStructInfo =
    runtime::TypedPackedFunc<StructInfo(const Call& call, const BlockBuilder& ctx)>;

/*!
 * \brief Packed function implementation for operators. The relax operator will be lowered to
 * this packed function call during codegen.
 */
using FCallPacked = String;

/*! \brief Attributes used in MaxPool2d operator */
struct MaxPool2dAttrs : public tvm::AttrsNode<MaxPool2dAttrs> {
  Array<PrimExpr> kernel_size;
  Array<PrimExpr> stride;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  TVM_DECLARE_ATTRS(MaxPool2dAttrs, "relax.attrs.MaxPool2dAttrs") {
    TVM_ATTR_FIELD(kernel_size).describe("The size of the window to take a max over.");
    TVM_ATTR_FIELD(stride).describe("The stride of the window.");
    TVM_ATTR_FIELD(padding).describe("The padding on the input.");
    TVM_ATTR_FIELD(dilation).describe("The stride of elements in the window.");
  }
};  // struct MaxPool2dAttrs

/*! \brief Attributes used in Conv2d operator */
struct Conv2dAttrs : public tvm::AttrsNode<Conv2dAttrs> {
  Array<PrimExpr> kernel_size;
  Array<PrimExpr> stride;
  Array<PrimExpr> padding;
  Array<PrimExpr> dilation;
  TVM_DECLARE_ATTRS(Conv2dAttrs, "relax.attrs.Conv2dAttrs") {
    TVM_ATTR_FIELD(kernel_size).describe("The size of the convolving kernel.");
    TVM_ATTR_FIELD(stride).describe("The stride of the convolution.");
    TVM_ATTR_FIELD(padding).describe("The padding on the input.");
    TVM_ATTR_FIELD(dilation).describe("The spacing between kernel elements.");
  }
};  // struct Conv2dAttrs

/*! \brief Attributes for dense operator */
struct DenseAttrs : public tvm::AttrsNode<DenseAttrs> {
  PrimExpr units;
  // tvm::String auto_scheduler_rewritten_layout;   // The layout after auto-scheduler's layout
  // rewrite Array<PrimExpr> meta_schedule_original_shape;  // The original shape of the weights
  DataType out_dtype;

  TVM_DECLARE_ATTRS(DenseAttrs, "relax.attrs.DenseAttrs") {
    TVM_ATTR_FIELD(units).describe("Number of hidden units of the dense transformation.");

    // use 0 bits to indicate none.
    TVM_ATTR_FIELD(out_dtype)
        .set_default(NullValue<DataType>())
        .describe("Output data type, set to explicit type under mixed precision setting");
  }
};

/*! \brief Attributes used in unique operator */
struct UniqueAttrs : public tvm::AttrsNode<UniqueAttrs> {
  bool sorted;
  bool return_inverse;
  bool return_counts;
  int dim;
  TVM_DECLARE_ATTRS(UniqueAttrs, "relax.attrs.UniqueAttrs") {
    TVM_ATTR_FIELD(sorted)
        .describe(
            "Whether to sort the unique elements in ascending order before returning as output.")
        .set_default(true);
    TVM_ATTR_FIELD(return_inverse)
        .describe(
            "Whether to return an additional tensor with indices for where elements in the "
            "original input ended up in the returned unique list.")
        .set_default(false);
    TVM_ATTR_FIELD(return_counts)
        .describe("Whether to return an additional tensor with counts of each unique elements")
        .set_default(false);
    TVM_ATTR_FIELD(dim)
        .describe(
            "The dimension to apply unique. If negative, the unique of the flattened input is "
            "returned.")
        .set_default(-1);
  }
};  // struct UniqueAttrs

struct PrintAttrs : public tvm::AttrsNode<PrintAttrs> {
  std::string format;
  TVM_DECLARE_ATTRS(PrintAttrs, "relax.attrs.PrintAttrs") {
    TVM_ATTR_FIELD(format)
        .describe("Python-style format string to use for displaying the input. Ignored if empty.")
        .set_default("");
  }
};

struct AssertOpAttrs : public tvm::AttrsNode<AssertOpAttrs> {
  std::string format;
  TVM_DECLARE_ATTRS(AssertOpAttrs, "relax.attrs.AssertOpAttrs") {
    TVM_ATTR_FIELD(format)
        .describe(
            "Python-style format string to use for displaying "
            "an error message if the assert fails. "
            "Ignored if empty.")
        .set_default("");
  }
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_OP_ATTR_TYPES_H_
