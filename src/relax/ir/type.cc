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

ObjectType::ObjectType(Span span) {
  ObjectPtr<ObjectTypeNode> n = make_object<ObjectTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ObjectTypeNode);

TVM_REGISTER_GLOBAL("relax.ObjectType").set_body_typed([](Span span) { return ObjectType(span); });

DynTensorType::DynTensorType(int ndim, DataType dtype, Span span) {
  ObjectPtr<DynTensorTypeNode> n = make_object<DynTensorTypeNode>();
  n->ndim = std::move(ndim);
  n->dtype = std::move(dtype);
  n->span = span;
  data_ = std::move(n);
}

DynTensorType DynTensorType::CreateUnknownNDim(DataType dtype, Span span) {
  ObjectPtr<DynTensorTypeNode> n = make_object<DynTensorTypeNode>();
  n->ndim = -1;
  n->dtype = std::move(dtype);
  n->span = std::move(span);
  return DynTensorType(std::move(n));
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

bool IsBaseOf(const Type& base, const Type& derived) {
  // TODO(@yuchen): refactor to use generic type functor, dispatching on base.

  if (auto base_tensor = base.as<DynTensorTypeNode>()) {
    if (auto derived_tensor = derived.as<DynTensorTypeNode>()) {
      if (base_tensor->IsUnknownNdim() || base_tensor->ndim == derived_tensor->ndim) {
        if (base_tensor->IsUnknownDtype() || base_tensor->dtype == derived_tensor->dtype) {
          return true;
        }
      }
    }
    return false;
  } else if (base.as<ShapeTypeNode>()) {
    if (derived.as<ShapeTypeNode>()) {
      return true;
    }
    return false;
  } else if (auto base_tuple = base.as<TupleTypeNode>()) {
    if (auto derived_tuple = derived.as<TupleTypeNode>()) {
      if (base_tuple->fields.size() != derived_tuple->fields.size()) {
        return false;
      }

      for (size_t i = 0; i < base_tuple->fields.size(); ++i) {
        if (!IsBaseOf(base_tuple->fields[i], derived_tuple->fields[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  } else if (auto base_func = base.as<FuncTypeNode>()) {
    if (auto derived_func = derived.as<FuncTypeNode>()) {
      if (base_func->arg_types.size() != derived_func->arg_types.size()) {
        return false;
      }
      for (size_t i = 0; i < base_func->arg_types.size(); ++i) {
        if (!IsBaseOf(base_func->arg_types[i], derived_func->arg_types[i])) {
          return false;
        }
      }
      if (!IsBaseOf(base_func->ret_type, derived_func->ret_type)) {
        return false;
      }
      return true;
    }
    return false;
  } else if (base.as<ObjectTypeNode>()) {
    return true;
  } else {
    LOG(FATAL) << "TypeError: cannot handle base type: " << base->GetTypeKey();
  }

  return false;
}

TVM_REGISTER_GLOBAL("relax.IsBaseOf").set_body_typed([](const Type& base, const Type& derived) {
  return IsBaseOf(base, derived);
});

}  // namespace relax
}  // namespace tvm
