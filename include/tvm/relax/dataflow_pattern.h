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
 * \file tvm/relax/dataflow_pattern.h
 * \brief A pattern language for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_PATTERN_H_
#define TVM_RELAX_DATAFLOW_PATTERN_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relay/dataflow_pattern.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace relax {

using relay::DFPattern;
using relay::DFPatternNode;

// class ShapeExprPattern; ? we don't need this right?
// class FunctionPattern;

// FIXME: Document those APIs.
class ExternFuncPattern;
class ExternFuncPatternNode : public DFPatternNode {
 public:
  String global_symbol_;
  const String& global_symbol() const { return global_symbol_; }
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("global_symbol", &global_symbol_); }

  static constexpr const char* _type_key = "relax.dataflow_pattern.ExternFuncPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExternFuncPatternNode, DFPatternNode);
};

class ExternFuncPattern : public DFPattern {
 public:
  TVM_DLL ExternFuncPattern(String global_symbol);
  TVM_DEFINE_OBJECT_REF_METHODS(ExternFuncPattern, DFPattern, ExternFuncPatternNode);
};

class VarPattern;
class VarPatternNode : public DFPatternNode {
 public:
  String name;
  const String& name_hint() const { return name; }
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  static constexpr const char* _type_key = "relax.dataflow_pattern.VarPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarPatternNode, DFPatternNode);
};

class VarPattern : public DFPattern {
 public:
  TVM_DLL VarPattern(String name_hint);
  TVM_DEFINE_OBJECT_REF_METHODS(VarPattern, DFPattern, VarPatternNode);
};

/*!
 * \brief A Pattern to Match a Relax Dataflow Variable
 */
class DataflowVarPattern;
/*! \brief Container for Var */
class DataflowVarPatternNode : public VarPatternNode {
 public:
  static constexpr const char* _type_key = "relax.dataflow_pattern.DataflowVarPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowVarPatternNode, DFPatternNode);
};

class DataflowVarPattern : public VarPattern {
 public:
  using VarPattern::VarPattern;
  TVM_DEFINE_OBJECT_REF_METHODS(DataflowVarPattern, VarPattern, DataflowVarPatternNode);
};

class DynTensorTypePattern;
class DynTensorTypePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  DynTensorType type;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("type", &type);
  }
  static constexpr const char* _type_key = "relax.dataflow_pattern.DynTensorTypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DynTensorTypePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern to match expressions with a specific dynamic tensor type.
 */
class DynTensorTypePattern : public DFPattern {
 public:
  TVM_DLL DynTensorTypePattern(DFPattern pattern, DynTensorType type);
  TVM_DEFINE_OBJECT_REF_METHODS(DynTensorTypePattern, DFPattern, DynTensorTypePatternNode);
};

class RuntimeDepShapePattern;
/*!
 * \brief Pattern for RuntimeDepShape.
 */
class RuntimeDepShapePatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relax.dataflow_pattern.RuntimeDepShapePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(RuntimeDepShapePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern to match expressions with runtime-dependent shapes.
 */
class RuntimeDepShapePattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RuntimeDepShapePattern, DFPattern, RuntimeDepShapePatternNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_PATTERN_H_
