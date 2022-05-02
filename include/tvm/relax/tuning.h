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
 * \file tvm/relax/tuning.h
 * \brief Relax Tuning Pass APIs.
 */
#ifndef TVM_RELAX_TRANSFORM_TUNING_H_
#define TVM_RELAX_TRANSFORM_TUNING_H_
#include <tvm/ir/module.h>

namespace tvm {
namespace relax {
class ChoiceNode : public runtime::Object {
 public:
  /*! \brief The function type of `f_transform` method. */
  using FTransform = runtime::TypedPackedFunc<void()>;
  /*! \brief The function type of `f_constr` method. */
  using FConstr = runtime::TypedPackedFunc<bool()>;
  /*! \brief transformation function */
  FTransform f_transform;
  /*! \brief constraint function.
     f_transform will be applied only when this function returns true. */
  FConstr f_constr;

  /*! \brief The default destructor. */
  virtual ~ChoiceNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_transform` is not visited
    // `f_constr` is not visited
  }

  FTransform GetTransformFunc() {
    ICHECK(f_transform != nullptr) << "Choice's f_transform method not implemented!";
    return f_transform;
  }

  FConstr GetConstrFunc() {
    ICHECK(f_constr != nullptr) << "Choice's f_constr method not implemented!";
    return f_constr;
  }

  static constexpr const char* _type_key = "relax.transform.Choice";
  TVM_DECLARE_BASE_OBJECT_INFO(ChoiceNode, Object);
};

class Choice : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Choice(ChoiceNode::FTransform f_transform, ChoiceNode::FConstr f_constr);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Choice, ObjectRef, ChoiceNode);
  // TVM_DEFINE_OBJECT_REF_METHODS(
  // TVM_DEFINE_OBJECT_REF_COW_METHOD(ChoiceNode);
};
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_TRANSFORM_TUNING_H_