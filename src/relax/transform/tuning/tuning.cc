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
 * \file src/relax/transform/tuning/tuning.cc
 * \brief Implementation of tuning APIs.
 */
#include <tvm/relax/tuning.h>

namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(ChoiceNode);

Choice::Choice(ChoiceNode::FTransform f_transform, ChoiceNode::FConstr f_constr) {
  ObjectPtr<ChoiceNode> n = make_object<ChoiceNode>();
  n->f_transform = f_transform;
  n->f_constr = f_constr;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.transform.Choice")
    .set_body_typed([](ChoiceNode::FTransform f_transform, ChoiceNode::FConstr f_constr) {
      return Choice(f_transform, f_constr);
    });

TVM_REGISTER_GLOBAL("relax.transform.ChoiceGetTransformFunc")
    .set_body_method<Choice>(&ChoiceNode::GetTransformFunc);

TVM_REGISTER_GLOBAL("relax.transform.ChoiceGetConstrFunc")
    .set_body_method<Choice>(&ChoiceNode::GetConstrFunc);

}  // namespace relax
}  // namespace tvm