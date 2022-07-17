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
 * \file tvm/relax/type.h
 * \brief Relax typed AST nodes.
 */
#ifndef TVM_RELAX_TYPE_H_
#define TVM_RELAX_TYPE_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/env_func.h>
#include <tvm/ir/tensor_type.h>
#include <tvm/ir/type.h>
#include <tvm/ir/type_relation.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>

#include <string>

namespace tvm {
namespace relax {

class ShapeTypeNode : public TypeNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("span", &span); }

  bool SEqualReduce(const ShapeTypeNode* other, SEqualReducer equal) const { return true; }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(0); }

  static constexpr const char* _type_key = "relax.ShapeType";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeTypeNode, TypeNode);
};

class ShapeType : public Type {
 public:
  TVM_DLL ShapeType(Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ShapeType, Type, ShapeTypeNode);
};

class ObjectTypeNode : public TypeNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("span", &span); }

  bool SEqualReduce(const ObjectTypeNode* other, SEqualReducer equal) const { return true; }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(0); }

  static constexpr const char* _type_key = "relax.ObjectType";
  TVM_DECLARE_FINAL_OBJECT_INFO(ObjectTypeNode, TypeNode);
};

class ObjectType : public Type {
 public:
  TVM_DLL ObjectType(Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ObjectType, Type, ObjectTypeNode);
};

class DynTensorTypeNode : public BaseTensorTypeNode {
 public:
  /*!
   * \brief The number of dimensions of the tensor, use -1 to denote tensor with unknwon number of
   * dimensions.
   */
  int ndim;
  /*! \brief The content data type, use void to denote the dtype is unknown. */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("ndim", &ndim);
    v->Visit("dtype", &dtype);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DynTensorTypeNode* other, SEqualReducer equal) const {
    return equal(ndim, other->ndim) && equal(dtype, other->dtype);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(ndim);
    hash_reduce(dtype);
  }

  inline bool IsUnknownNdim() const { return ndim == -1; }

  inline bool IsUnknownDtype() const { return dtype.is_void(); }

  static constexpr const char* _type_key = "relax.DynTensorType";
  TVM_DECLARE_FINAL_OBJECT_INFO(DynTensorTypeNode, BaseTensorTypeNode);
};

/*!
 * \brief Managed reference to DynTensorTypeNode.
 * \sa DynTensorTypeNode.
 */
class DynTensorType : public Type {
 public:
  /*!
   * \brief Constructor.
   * \param shape The shape of the tensor.
   * \param dtype The runtime dtype of the tensor's elements.
   */
  TVM_DLL DynTensorType(int ndim, DataType dtype, Span span = Span());

  /*!
   * \brief Create a DynTensorType with unknown ndim.
   */
  TVM_DLL static DynTensorType CreateUnknownNDim(DataType dtype, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(DynTensorType, Type, DynTensorTypeNode);
};

class DimTypeNode : public TypeNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("span", &span); }

  bool SEqualReduce(const DimTypeNode* other, SEqualReducer equal) const { return true; }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(0); }

  static constexpr const char* _type_key = "relax.DimType";
  TVM_DECLARE_FINAL_OBJECT_INFO(DimTypeNode, TypeNode);
};

class DimType : public Type {
 public:
  TVM_DLL DimType(Span span = Span());

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DimType, Type, DimTypeNode);
};

/*!
 * \brief Check the subtype relationship between base and derived.
 * \param base The base type.
 * \param derived The derived type.
 * \return If \p derived is a subtype of \p base or if both are the same type, returns true.
 * Otherwise returns false.
 */
bool IsBaseOf(const Type& base, const Type& derived);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TYPE_H_
