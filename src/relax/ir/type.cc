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
 * \file src/relax/type.cc
 * \brief Relax's type system AST nodes throughout the IR.
 */
#include <tvm/relax/type.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(ShapeTypeNode);

ShapeType::ShapeType(Span span) {
  ObjectPtr<ShapeTypeNode> n = make_object<ShapeTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ShapeType").set_body_typed([](Span span) { return ShapeType(span); });

DynTensorType::DynTensorType(int ndim, DataType dtype, Span span) {
  ObjectPtr<DynTensorTypeNode> n = make_object<DynTensorTypeNode>();
  n->ndim = std::move(ndim);
  n->dtype = std::move(dtype);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DynTensorTypeNode);

TVM_REGISTER_GLOBAL("relax.DynTensorType").set_body_typed([](int ndim, DataType dtype, Span span) {
  return DynTensorType(ndim, dtype, span);
});

DimType::DimType(Span span) {
  ObjectPtr<DimTypeNode> n = make_object<DimTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DimTypeNode);

TVM_REGISTER_GLOBAL("relax.DimType").set_body_typed([](Span span) { return DimType(span); });

bool IsSubType(Type lhs, Type rhs) {
  if (auto lhs_tensor = lhs.as<DynTensorTypeNode>()) {
    if (auto rhs_tensor = rhs.as<DynTensorTypeNode>()) {
      if (rhs_tensor->ndim == -1 || rhs_tensor->ndim == lhs_tensor->ndim) {
        if (rhs_tensor->dtype == DataType::Void() || rhs_tensor->dtype == lhs_tensor->dtype) {
          return true;
        }
      }
    }
  }

  if (lhs->IsInstance<ShapeTypeNode>() && rhs->IsInstance<ShapeTypeNode>()) {
    return true;
  }

  if (auto lhs_tuple = lhs.as<TupleTypeNode>()) {
    if (auto rhs_tuple = rhs.as<TupleTypeNode>()) {
      if (lhs_tuple->fields.size() != rhs_tuple->fields.size()) {
        return false;
      }

      for (size_t i = 0; i < lhs_tuple->fields.size(); i++) {
        if (!IsSubType(lhs_tuple->fields[i], rhs_tuple->fields[i])) {
          return false;
        }
      }
      return true;
    }
  }
  // TODO(@yuchen): consider ObjectType relation after it's merged.

  return false;
}

TVM_REGISTER_GLOBAL("relax.IsSubType").set_body_typed([](Type lhs, Type rhs) {
  return IsSubType(lhs, rhs);
});

}  // namespace relax
}  // namespace tvm
