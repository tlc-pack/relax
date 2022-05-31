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

#include "../../../meta_schedule/utils.h"
namespace tvm {
namespace relax {
TVM_REGISTER_NODE_TYPE(ChoiceNode);

Choice::Choice(FTransform f_transform, FConstr f_constr) {
  ICHECK(f_transform != nullptr) << "Transformation function should be defined.";
  if (f_constr == nullptr) {
    f_constr = [=](IRModule m) { return true; };
  }

  ObjectPtr<ChoiceNode> n = make_object<ChoiceNode>();
  n->f_transform = std::move(f_transform);
  n->f_constr = std::move(f_constr);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.transform.Choice")
    .set_body_typed([](FTransform f_transform, FConstr f_constr) {
      return Choice(f_transform, f_constr);
    });

TVM_REGISTER_GLOBAL("relax.transform.ChoiceGetTransformFunc")
    .set_body_method<Choice>(&ChoiceNode::GetTransformFunc);

TVM_REGISTER_GLOBAL("relax.transform.ChoiceGetConstrFunc")
    .set_body_method<Choice>(&ChoiceNode::GetConstrFunc);
TVM_REGISTER_GLOBAL("relax.transform.ChoiceCheckConstr")
    .set_body_method<Choice>(&ChoiceNode::CheckConstr);

TVM_REGISTER_NODE_TYPE(KnobNode);
Knob::Knob(String name, Map<String, Choice> choices) {
  ObjectPtr<KnobNode> n = make_object<KnobNode>();
  n->name = std::move(name);
  n->choices = std::move(choices);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.transform.Knob")
    .set_body_typed([](String name, Map<String, Choice> choices) { return Knob(name, choices); });

TVM_REGISTER_GLOBAL("relax.transform.KnobVerify").set_body_method<Knob>(&KnobNode::Verify);
TVM_REGISTER_GLOBAL("relax.transform.KnobApply").set_body_method<Knob>(&KnobNode::Apply);

TVM_REGISTER_NODE_TYPE(TraceNode);
Trace::Trace(IRModule in_mod, Array<Knob> knobs, Array<String> decisions) {
  ICHECK(knobs.size() == decisions.size()) << "Size of knobs and decisions should match";
  // Deep-copy IRModule
  IRModule out_mod = meta_schedule::DeepCopyIRModule(in_mod);
  // Apply the decision history if provided
  int size = knobs.size();
  for (int i = 0; i < size; i++) {
    out_mod = knobs[i]->Apply(out_mod, decisions[i]);
  }

  ObjectPtr<TraceNode> n = make_object<TraceNode>();
  n->in_mod = std::move(in_mod);
  n->out_mod = std::move(out_mod);
  n->knobs = std::move(knobs);
  n->decisions = std::move(decisions);
  n->size = std::move(size);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.transform.Trace")
    .set_body_typed([](IRModule in_mod, Array<Knob> knobs, Array<String> decisions) {
      return Trace(in_mod, knobs, decisions);
    });
TVM_REGISTER_GLOBAL("relax.transform.TraceVerify").set_body_method<Trace>(&TraceNode::Verify);
TVM_REGISTER_GLOBAL("relax.transform.TraceAdd").set_body_method<Trace>(&TraceNode::Add);
TVM_REGISTER_GLOBAL("relax.transform.TraceSetPerf").set_body_method<Trace>(&TraceNode::SetPerf);
}  // namespace relax
}  // namespace tvm
