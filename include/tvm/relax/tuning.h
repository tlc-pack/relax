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
#ifndef TVM_RELAX_TUNING_H_
#define TVM_RELAX_TUNING_H_
#include <tvm/ir/module.h>

namespace tvm {
namespace relax {
/*! \brief The function type of `f_transform` method. */
using FTransform = runtime::TypedPackedFunc<IRModule(IRModule)>;
/*! \brief The function type of `f_constr` method. */
using FConstr = runtime::TypedPackedFunc<bool(IRModule)>;

/*! \brief Choice manages a set of transformation and constraint functions. */
class ChoiceNode : public runtime::Object {
 public:
  /*! \brief transformation function. */
  FTransform f_transform;
  /*! \brief constraint function.
     f_transform will be applied only when this function returns true. */
  FConstr f_constr;

  /*! \brief The default destructor. */
  virtual ~ChoiceNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `f_transform` is not visited.
    // `f_constr` is not visited.
  }

  /*! \brief Getter for f_transform. */
  FTransform GetTransformFunc() {
    ICHECK(f_transform != nullptr) << "Choice's f_transform method not implemented!";
    return f_transform;
  }

  /*! \brief Getter for f_constr. */
  FConstr GetConstrFunc() {
    ICHECK(f_constr != nullptr) << "Choice's f_constr method not implemented!";
    return f_constr;
  }

  /*! \brief Perform f_constr. */
  bool CheckConstr(IRModule mod) { return f_constr(mod); }

  static constexpr const char* _type_key = "relax.transform.Choice";
  TVM_DECLARE_BASE_OBJECT_INFO(ChoiceNode, Object);
};

/*! \brief Managed reference to ChoiceNode */
class Choice : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Choice(FTransform f_transform, FConstr f_constr);

  // TODO(sunggg): Double-check this
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Choice, ObjectRef, ChoiceNode);
};

/*! \brief Knob manages a set of valid choices for an optimization. */
class KnobNode : public runtime::Object {
 public:
  /*! \brief Name of the knob. */
  String name;
  /*! \brief Decision space. */
  Map<String, Choice> choices;

  /*! \brief The default destructor. */
  virtual ~KnobNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("choices", &choices);
  }

  /*! \brief Check if a decision is valid. */
  bool Verify(String decision) { return choices.count(decision) > 0; }

  /*! \brief Apply decision. */
  IRModule Apply(IRModule mod, String decision) {
    ICHECK(Verify(decision)) << "Invalid choice for this knob: " << decision << "\n";
    ICHECK(choices[decision]->CheckConstr(mod)) << "Constraint is not satisfied.\n";
    return choices[decision]->f_transform(mod);
  }

  static constexpr const char* _type_key = "relax.transform.Knob";
  TVM_DECLARE_BASE_OBJECT_INFO(KnobNode, Object);
};

/*! \brief Managed reference to KnobNode */
class Knob : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Knob(String name, Map<String, Choice> choices);

  // TODO(sunggg): Double-check this
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Knob, ObjectRef, KnobNode);
};

/*! \brief Trace manages history of optimization decisions. */
class TraceNode : public runtime::Object {
 public:
  /*! \brief Input IRModule. */
  IRModule in_mod;
  /*! \brief Output IRModule. */
  mutable IRModule out_mod;
  // TODO(sunggg): can we move knobs and decisions into private?
  /*! \brief Knobs that are applied so far. */
  Array<Knob> knobs;
  /*! \brief Decisions made for the knobs. */
  Array<String> decisions;
  /*! \brief Performance of out_mod. */
  // TODO(sunggg): Match with MetaSchedule
  mutable double perf = -1;
  /*! \brief Length of the decision history. */
  mutable int size = 0;
  /*! \brief The default destructor. */
  virtual ~TraceNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("in_mod", &in_mod);
    v->Visit("out_mod", &out_mod);
    v->Visit("knobs", &knobs);
    v->Visit("decisions", &decisions);
    v->Visit("perf", &perf);
    v->Visit("size", &size);
  }

  /*! \brief Verify current decision history. */
  bool Verify() {
    if (knobs.size() != decisions.size()) return false;
    int n = knobs.size();
    for (int i = 0; i < n; i++) {
      if (!knobs[i]->Verify(decisions[i])) return false;
    }
    return true;
  }

  /*! \brief Add a knob and its decision to the current trace. */
  IRModule Add(Knob knob, String decision) {
    out_mod = knob->Apply(out_mod, decision);
    knobs.push_back(knob);
    decisions.push_back(decision);
    // perf number should be initialized after new decision is applied.
    perf = -1;
    // increment history size.
    size++;
    return out_mod;
  }

  /*! \brief Set the performance. */
  void SetPerf(double _perf) { perf = _perf; }

  static constexpr const char* _type_key = "relax.transform.Trace";
  TVM_DECLARE_BASE_OBJECT_INFO(TraceNode, Object);
};

/*! \brief Managed reference to TraceNode */
class Trace : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Trace(IRModule in_mod, Array<Knob> knobs, Array<String> decisions);

  // TODO(sunggg): Double-check this
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Trace, ObjectRef, TraceNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TUNING_H_
