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
 * \file tvm/relax/attrs/shape.h
 * \brief Attributes for shape operators.
 */
#ifndef TVM_RELAX_ATTRS_SHAPE_H_
#define TVM_RELAX_ATTRS_SHAPE_H_

#include <tvm/ir/attrs.h>

namespace tvm {
namespace relax {
/*!
 * \brief Attributes for decoding/making shape.
 */
struct ShapeAttrs : public tvm::AttrsNode<ShapeAttrs> {
  Array<Integer> indices;

  TVM_DECLARE_ATTRS(ShapeAttrs, "relax.attrs.ShapeAttrs") {
    TVM_ATTR_FIELD(indices).describe("The indices of the heap to decode/make the shape to/from.");
  }
};

// /*!
//  * \brief Attributes for making shape.
//  */
// struct MakeShapeAttrs : public tvm::AttrsNode<MakeShapeAttrs> {
//   Array<Integer> indices;

//   TVM_DECLARE_ATTRS(MakeShapeAttrs, "relax.attrs.MakeShapeAttrs") {
//     TVM_ATTR_FIELD(indices).describe("The indices of the heap to make the shape from.");
//   }
// };

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_ATTRS_SHAPE_H_
