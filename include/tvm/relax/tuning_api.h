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
 * \file tvm/relax/tuning_api.h
 * \brief Relax Tuning Pass APIs.
 */
#ifndef TVM_RELAX_TUNING_API_H_
#define TVM_RELAX_TUNING_API_H_
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/meta_schedule/database.h>

namespace tvm {
namespace relax {

/*! \brief Helper function to unpack arguments in the array as parameters for the given packed
 * function. */
TVM_ALWAYS_INLINE TVMRetValue CallPackedWithArgsInArray(const runtime::PackedFunc f,
                                                        const Array<ObjectRef>& args) {
  const size_t kNumArgs = args.size();
  TVMValue tvm_values[kNumArgs];
  int tvm_type_codes[kNumArgs];
  runtime::TVMArgsSetter setter(tvm_values, tvm_type_codes);
  const ObjectRef* ptr = args.template as<ArrayNode>()->begin();
  for (size_t i = 0; i < kNumArgs; ++i) {
    setter(i, *(ptr + i));
  }

  TVMRetValue rv;
  f.CallPacked(TVMArgs(tvm_values, tvm_type_codes, kNumArgs), &rv);
  return rv;
}

/*! \brief Choice manages a set of keys for transformation and constraint functions. */
class ChoiceNode : public runtime::Object {
 public:
  /*! \brief key for transformation function. */
  String f_transform_key;
  /*! \brief key for constraint function.
     f_transform will be applied only when this function returns true. */
  String f_constr_key;
  Array<ObjectRef> f_transform_args;
  Array<ObjectRef> f_constr_args;

  /*! \brief The default destructor. */
  virtual ~ChoiceNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("f_transform_key", &f_transform_key);
    v->Visit("f_transform_args", &f_transform_args);
    v->Visit("f_constr_key", &f_constr_key);
    v->Visit("f_constr_args", &f_constr_args);
  }

  /*! \brief Getter for f_constr. */
  const runtime::PackedFunc GetConstrFunc() {
    const auto* f_constr = tvm::runtime::Registry::Get(f_constr_key);
    ICHECK(f_constr != nullptr) << "f_constr_key is not registered: " << f_constr_key;
    return *f_constr;
  }

  /*! \brief Getter for f_transform. */
  const runtime::PackedFunc GetTransformFunc() {
    auto* f_transform = tvm::runtime::Registry::Get(f_transform_key);
    ICHECK(f_transform != nullptr) << "f_transform_key is not registered: " << f_transform_key;
    return *f_transform;
  }

  /*! \brief Perform f_constr. */
  bool CheckConstr(IRModule mod) {
    Array<ObjectRef> args(f_constr_args);
    args.insert(args.begin(), mod);
    return CallPackedWithArgsInArray(GetConstrFunc(), args);
  }

  /*! \brief Perform f_transform. */
  IRModule ApplyTransformFunc(IRModule mod) {
    // Apply transformation when constraint is satisfied.
    if (CheckConstr(mod)) {
      Array<ObjectRef> args(f_transform_args);
      args.insert(args.begin(), GetRef<IRModule>(mod.CopyOnWrite()));
      return CallPackedWithArgsInArray(GetTransformFunc(), args);
    }
    return mod;
  }

  /*!
   * \brief Serialize Choice as a JSON-style object
   * \return The JSON-style object
   */
  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "relax.tuning_api.Choice";
  TVM_DECLARE_BASE_OBJECT_INFO(ChoiceNode, Object);
};

/*! \brief Managed reference to ChoiceNode */
class Choice : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Choice(String f_transform_key, Array<ObjectRef> f_transform_args,
                          String f_constr_key, Array<ObjectRef> f_constr_args);
  /*! \brief Deserialize JSON-style object into Choice */
  TVM_DLL static Choice FromJSON(const ObjectRef& json_obj);
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

  /*! \brief Apply decision if the constraint is satisfied.
      Otherwise, return the original IRModule.
   */
  IRModule Apply(IRModule mod, String decision) {
    ICHECK(Verify(decision)) << "Invalid choice for this knob: " << decision << "\n";
    return choices[decision]->ApplyTransformFunc(mod);
  }

  /*!
   * \brief Serialize Knob as a JSON-style object
   * \return The JSON-style object
   */
  ObjectRef AsJSON() const;

  static constexpr const char* _type_key = "relax.tuning_api.Knob";
  TVM_DECLARE_BASE_OBJECT_INFO(KnobNode, Object);
};

/*! \brief Managed reference to KnobNode */
class Knob : public runtime::ObjectRef {
 public:
  TVM_DLL explicit Knob(String name, Map<String, Choice> choices);
  /*! \brief Deserialize JSON-style object into Knob */
  TVM_DLL static Knob FromJSON(const ObjectRef& json_obj);
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
  bool Verify() const {
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

  /*!
   * \brief Serialize Trace as a JSON-style object
   * \param include_irmod Boolean config to include IRModules in the output.
   * \return The JSON-style object
   */
  ObjectRef AsJSON(bool include_irmod = true) const;

  /*! \brief Set the performance. */
  void SetPerf(double _perf) { perf = _perf; }
  /*! \brief Set output module. */
  void SetOutMod(IRModule mod_) { out_mod = mod_; }

  static constexpr const char* _type_key = "relax.tuning_api.Trace";
  TVM_DECLARE_BASE_OBJECT_INFO(TraceNode, Object);
};

/*! \brief Managed reference to TraceNode */
class Trace : public runtime::ObjectRef {
 public:
  /*! \brief Default constructor. Creating an empty trace. */
  Trace();
  /*!
   * \brief Constructor. Creating a trace from existing knobs and their decisions
   * \param in_mod Input IRModule
   * \param knobs The knobs used
   * \param decisions The decisions made in sampling
   */
  TVM_DLL explicit Trace(IRModule in_mod, Array<Knob> knobs, Array<String> decisions);
  /*! \brief Deserialize JSON-style object into Trace */
  TVM_DLL static Trace FromJSON(const ObjectRef& json_obj);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Trace, ObjectRef, TraceNode);
};

/*! \brief The class of tuning records. */
class TuningRecordNode : public runtime::Object {
 public:
  /*! \brief The trace tuned. */
  Trace trace;
  /*! \brief The measurement record in seconds. */
  Optional<Array<FloatImm>> run_secs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("trace", &trace);
    v->Visit("run_secs", &run_secs);
  }

  static constexpr const char* _type_key = "relax.tuning_api.TuningRecord";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuningRecordNode, runtime::Object);

  /*!
   * \brief Export the tuning record to a JSON string.
   * \param include_irmod Boolean config to include IRModules in the output.
   * \return JSON object
   */
  ObjectRef AsJSON(bool include_irmod = false) const;
};

/*!
 * \brief The managed reference of TuningRecordNode.
 * \sa TuningRecordNode
 */
class TuningRecord : public runtime::ObjectRef {
 public:
  /*!
   \brief Constructor of a tuning record.
   \param trace The trace of the tuning record.
   \param run_secs The running time of the tuning record.
  */
  TVM_DLL explicit TuningRecord(Trace trace, Optional<Array<FloatImm>> run_secs);
  /*!
   * \brief Create a tuning record from a json object.
   * \param json_obj The json object.
   * \param workload The workload.
   * \return The tuning record created.
   */
  TVM_DLL static TuningRecord FromJSON(const ObjectRef& json_obj);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TuningRecord, runtime::ObjectRef, TuningRecordNode);
};

/* \brief The abstract interface of database. */
class DatabaseNode : public runtime::Object {
 public:
  /*! \brief Default destructor */
  virtual ~DatabaseNode() = default;
  /*!
   * \brief Check if the database has the given workload.
   * \param mod The IRModule to be searched for.
   * \return Whether the database has the given workload.
   */
  virtual bool HasWorkload(const IRModule& mod) = 0;
  /*!
   * \brief Check if the database has a measurement record for the given workload and target pair.
   * \param workload The workload to be searched for.
   * \param target The target to be searched for.
   * \return Whether the database has the measurement record for given workload and target pair.
   */
  virtual bool HasMeasurementRecord(const meta_schedule::Workload& workload,
                                    const Target& target) = 0;
  /*!
   * \brief Check if the database has a tuning record for the given workload and target pair.
   * \param workload The workload to be searched for.
   * \param target The target to be searched for.
   * \return Whether the database has the tuning record for the given workload and target pair.
   */
  virtual bool HasTuningRecord(const meta_schedule::Workload& workload, const Target& target) = 0;
  /*!
   * \brief Look up or add workload to the database if missing.
   * \param mod The IRModule to be searched for or added.
   * \return The workload corresponding to the given IRModule.
   */
  virtual meta_schedule::Workload CommitWorkload(const IRModule& mod) = 0;
  /*!
   * \brief Add a measurement record for a given pair of target and workload to the database.
   * \param workload Workload to be searched for.
   * \param target Target to be searched for.
   * \param record Measurement record to be added.
   */
  virtual void CommitMeasurementRecord(const meta_schedule::Workload& workload,
                                       const Target& target, const Array<FloatImm>& record) = 0;
  /*!
   * \brief Add a tuning record for a given pair of target and workload to the database.
   * \param workload Workload to be searched for.
   * \param target Target to be searched for.
   * \param record Tuning record to be added.
   */
  virtual void CommitTuningRecord(const meta_schedule::Workload& workload, const Target& target,
                                  const TuningRecord& record) = 0;
  /*!
   * \brief Get the top K tuning records of given workload and target from the database.
   * \param workload The workload to be searched for.
   * \param target Target to be searched for.
   * \param top_k The number of top records to be returned.
   * \return An array of top K tuning records for the given workload.
   */
  virtual Array<TuningRecord> GetTopK(const meta_schedule::Workload& workload, const Target& target,
                                      int top_k) = 0;
  /*!
   * \brief Get the measurement record of given workload and target from the database.
   * \param workload The workload to be searched for.
   * \param target Target to be searched for.
   * \return Measurement.
   */
  virtual Array<FloatImm> GetMeasurementRecord(const meta_schedule::Workload& workload,
                                               const Target target) = 0;

  static constexpr const char* _type_key = "relax.tuning_api.Database";
  TVM_DECLARE_BASE_OBJECT_INFO(DatabaseNode, runtime::Object);
};

/*!
 * \brief Managed reference to DatabaseNode.
 * \sa DatabaseNode
 */
class Database : public runtime::ObjectRef {
 public:
  /*!
   * \brief Create a default database that uses JSON file for tuning records.
   * \param path_workload The path to the workload table.
   * \param path_tuning_record The path to the tuning record table.
   * \param path_measurement_record The path to the measurement_record table.
   * \param allow_missing Whether to create new file when the given path is not found.
   */
  TVM_DLL static Database JSONDatabase(String path_workload, String path_tuning_record,
                                       String path_measurement_record, bool allow_missing);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Database, runtime::ObjectRef, DatabaseNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TUNING_API_H_
