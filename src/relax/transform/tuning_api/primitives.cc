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
 * \file src/relax/transform/tuning_api/primitives.cc
 * \brief Implementation of tuning APIs.
 */

#include <tvm/relax/tuning_api.h>

#include "../../../meta_schedule/utils.h"
namespace tvm {
namespace relax {

Choice::Choice(String f_transform_key, Array<ObjectRef> f_transform_args, String f_constr_key,
               Array<ObjectRef> f_constr_args) {
  ObjectPtr<ChoiceNode> n = make_object<ChoiceNode>();
  n->f_transform_key = std::move(f_transform_key);
  n->f_transform_args = std::move(f_transform_args);
  n->f_constr_key = std::move(f_constr_key);
  n->f_constr_args = std::move(f_constr_args);
  data_ = std::move(n);
}

// TODO (sunggg): Currently, it only supports an array of primitive data types.
ObjectRef ChoiceNode::AsJSON() const {
  Array<ObjectRef> json_transfrom_args, json_constr_args;
  for (ObjectRef arg : this->f_transform_args) {
    std::string json_arg = tvm::SaveJSON(arg);
    std::string b64_arg = meta_schedule::Base64Encode(json_arg);
    json_transfrom_args.push_back(String(b64_arg));
  }
  for (ObjectRef arg : this->f_constr_args) {
    std::string json_arg = tvm::SaveJSON(arg);
    std::string b64_arg = meta_schedule::Base64Encode(json_arg);
    json_constr_args.push_back(String(b64_arg));
  }
  return Array<ObjectRef>{
      this->f_transform_key,
      json_transfrom_args,
      this->f_constr_key,
      json_constr_args,
  };
}

Choice Choice::FromJSON(const ObjectRef& json) {
  // Parse `json` into `choice`
  String f_transform_key, f_constr_key;
  Array<ObjectRef> f_transform_args, f_constr_args;
  try {
    const ArrayNode* arr = json.as<ArrayNode>();
    ICHECK(arr && arr->size() == 4);
    const auto* arr0 = arr->at(0).as<StringObj>();
    const auto* arr1 = arr->at(1).as<ArrayNode>();
    const auto* arr2 = arr->at(2).as<StringObj>();
    const auto* arr3 = arr->at(3).as<ArrayNode>();
    ICHECK(arr0 && arr1 && arr2 && arr3);
    f_transform_key = GetRef<String>(arr0);
    {
      f_transform_args.reserve(arr1->size());
      for (const ObjectRef& elem : *arr1) {
        String b64_arg = Downcast<String>(elem);
        std::string json_arg = meta_schedule::Base64Decode(b64_arg);
        ObjectRef arg = LoadJSON(json_arg);
        f_transform_args.push_back(arg);
      }
    }
    f_constr_key = GetRef<String>(arr2);
    {
      f_constr_args.reserve(arr3->size());
      for (const ObjectRef& elem : *arr3) {
        String b64_arg = Downcast<String>(elem);
        std::string json_arg = meta_schedule::Base64Decode(b64_arg);
        ObjectRef arg = LoadJSON(json_arg);
        f_constr_args.push_back(arg);
      }
    }

  } catch (const tvm::Error& e) {
    LOG(FATAL)
        << "ValueError: The json entry of a choice should contain a set of two strings, but gets: "
        << json;
    throw;
  }
  return Choice(f_transform_key, f_transform_args, f_constr_key, f_constr_args);
}

Knob::Knob(String name, Map<String, Choice> choices) {
  ObjectPtr<KnobNode> n = make_object<KnobNode>();
  n->name = std::move(name);
  n->choices = std::move(choices);
  data_ = std::move(n);
}

ObjectRef KnobNode::AsJSON() const {
  Map<String, ObjectRef> json_choices;
  for (auto const& x : choices) {
    json_choices.Set(x.first, x.second->AsJSON());
  }
  return Array<ObjectRef>{
      /* 0: name    */ std::move(name),
      /* 1: choices */ std::move(json_choices),
  };
}

Knob Knob::FromJSON(const ObjectRef& json) {
  // Parse `json` into `name` and `choices`
  String name;
  Map<String, Choice> choices;
  try {
    const ArrayNode* arr = json.as<ArrayNode>();
    ICHECK(arr && arr->size() == 2);
    const auto* arr0 = arr->at(0).as<StringObj>();
    const auto* arr1 = arr->at(1).as<MapNode>();
    ICHECK(arr0 && arr1);
    name = GetRef<String>(arr0);
    for (auto const& x : GetRef<Map<String, ObjectRef>>(arr1)) {
      String decision = x.first;
      Choice choice = Choice::FromJSON(x.second);
      choices.Set(decision, choice);
    }
  } catch (const tvm::Error& e) {
    LOG(FATAL)
        << "ValueError: The json entry of a choice should contain a set of two strings, but gets: "
        << json;
    throw;
  }
  return Knob(name, choices);
}

Trace::Trace() { data_ = make_object<TraceNode>(); }

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

ObjectRef TraceNode::AsJSON() const {
  ICHECK(this->Verify()) << "Trace should be valid";
  std::string json_mod = tvm::SaveJSON(this->in_mod);
  std::string b64_mod = meta_schedule::Base64Encode(json_mod);

  Array<ObjectRef> json_knobs;
  Array<ObjectRef> json_decisions;

  int size = this->size;
  json_knobs.reserve(size);
  json_decisions.reserve(size);

  for (int i = 0; i < size; i++) {
    const Knob& knob = this->knobs[i];
    const String& decision = this->decisions[i];

    json_knobs.push_back(knob->AsJSON());
    json_decisions.push_back(decision);
  }

  return Array<ObjectRef>{String(b64_mod), json_knobs, json_decisions};
}

Trace Trace::FromJSON(const ObjectRef& json) {
  // Parse `json` into `trace`
  IRModule in_mod;
  Array<Knob> knobs;
  Array<String> decisions;
  try {
    const ArrayNode* arr = json.as<ArrayNode>();
    ICHECK(arr && arr->size() == 3);
    const auto* arr0 = arr->at(0).as<StringObj>();
    const auto* arr1 = arr->at(1).as<ArrayNode>();
    const auto* arr2 = arr->at(2).as<ArrayNode>();
    ICHECK(arr0 && arr1 && arr2);

    String b64_mod = GetRef<String>(arr0);
    std::string json_mod = meta_schedule::Base64Decode(b64_mod);
    in_mod = Downcast<IRModule>(LoadJSON(json_mod));

    for (const ObjectRef& elem : *arr1) {
      knobs.push_back(Knob::FromJSON(elem));
    }

    for (const ObjectRef& elem : *arr2) {
      decisions.push_back(Downcast<String>(elem));
    }
  } catch (const tvm::Error& e) {
    LOG(FATAL)
        << "ValueError: The json entry of a choice should contain a set of two strings, but gets: "
        << json;
    throw;
  }
  return Trace(in_mod, knobs, decisions);
}

/**************** FFI ****************/
TVM_REGISTER_NODE_TYPE(ChoiceNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.Choice")
    .set_body_typed([](String f_transform_key, Array<ObjectRef> f_transform_args,
                       String f_constr_key, Array<ObjectRef> f_constr_args) {
      return Choice(f_transform_key, f_transform_args, f_constr_key, f_constr_args);
    });
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceAsJSON").set_body_method<Choice>(&ChoiceNode::AsJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceFromJSON").set_body_typed(Choice::FromJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceGetTransformFunc")
    .set_body_method<Choice>(&ChoiceNode::GetTransformFunc);
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceGetConstrFunc")
    .set_body_method<Choice>(&ChoiceNode::GetConstrFunc);
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceApplyTransformFunc")
    .set_body_method<Choice>(&ChoiceNode::ApplyTransformFunc);
TVM_REGISTER_GLOBAL("relax.tuning_api.ChoiceCheckConstr")
    .set_body_method<Choice>(&ChoiceNode::CheckConstr);

TVM_REGISTER_NODE_TYPE(KnobNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.Knob")
    .set_body_typed([](String name, Map<String, Choice> choices) { return Knob(name, choices); });
TVM_REGISTER_GLOBAL("relax.tuning_api.KnobAsJSON").set_body_method<Knob>(&KnobNode::AsJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.KnobFromJSON").set_body_typed(Knob::FromJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.KnobVerify").set_body_method<Knob>(&KnobNode::Verify);
TVM_REGISTER_GLOBAL("relax.tuning_api.KnobApply").set_body_method<Knob>(&KnobNode::Apply);

TVM_REGISTER_NODE_TYPE(TraceNode);
TVM_REGISTER_GLOBAL("relax.tuning_api.Trace")
    .set_body_typed([](IRModule in_mod, Array<Knob> knobs, Array<String> decisions) {
      return Trace(in_mod, knobs, decisions);
    });
TVM_REGISTER_GLOBAL("relax.tuning_api.TraceVerify").set_body_method<Trace>(&TraceNode::Verify);
TVM_REGISTER_GLOBAL("relax.tuning_api.TraceAdd").set_body_method<Trace>(&TraceNode::Add);
TVM_REGISTER_GLOBAL("relax.tuning_api.TraceSetPerf").set_body_method<Trace>(&TraceNode::SetPerf);
TVM_REGISTER_GLOBAL("relax.tuning_api.TraceSetOutMod")
    .set_body_method<Trace>(&TraceNode::SetOutMod);

TVM_REGISTER_GLOBAL("relax.tuning_api.TraceAsJSON").set_body_method<Trace>(&TraceNode::AsJSON);
TVM_REGISTER_GLOBAL("relax.tuning_api.TraceFromJSON").set_body_typed(Trace::FromJSON);
}  // namespace relax
}  // namespace tvm