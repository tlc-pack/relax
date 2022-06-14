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

#include <tvm/ir/expr.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/container/optional.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace relax {

// FIXME: Document those APIs.
class DFPatternNode : public Object {
 public:
  static constexpr const char* _type_key = "DFPatternNode";
  TVM_DECLARE_BASE_OBJECT_INFO(DFPatternNode, Object);
};

class DFPattern : public ObjectRef {
 public:
  template <typename... Args>
  DFPattern operator()(Args&&... args) const;
  /*! \brief Syntatic Sugar for creating a CallPattern */
  DFPattern operator()(const std::vector<DFPattern>& args) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with an "add" op */
  DFPattern operator+(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "subtract" op */
  DFPattern operator-(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "multiply" op */
  DFPattern operator*(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating a CallPattern with a "divide" op */
  DFPattern operator/(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an OrPattern */
  DFPattern operator||(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an OrPattern */
  DFPattern operator&&(const DFPattern& other) const;
  /*! \brief Syntatic Sugar for creating an Optional Pattern */
  DFPattern Optional(const std::function<DFPattern(const DFPattern&)>& func) const;
  /*! \brief Syntatic Sugar for creating an AttrPattern */
  DFPattern HasAttr(const Map<String, ObjectRef>& attrs) const;
  /*! \brief Syntatic Sugar for creating a TypePattern */
  DFPattern HasType(const Type& type) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a DataType */
  DFPattern HasDtype(const DataType& dtype) const;
  /*! \brief Syntatic Sugar for creating a DataTypePattern with a data type's name */
  DFPattern HasDtype(const std::string& dtype) const;
  /*! \brief Syntatic Sugar for creating a ShapePattern */
  DFPattern HasShape(const Array<PrimExpr>& shape) const;
  /*! \brief Syntatic Sugar for creating a RuntimeDepShapePattern */
  DFPattern HasRuntimeDepShape() const;

  TVM_DEFINE_OBJECT_REF_METHODS(DFPattern, ObjectRef, DFPatternNode);
};

class ExprPatternNode : public DFPatternNode {
 public:
  /*! \brief The expression to match. */
  Expr expr;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("expr", &expr); }

  static constexpr const char* _type_key = "relax.dataflow_pattern.ExprPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExprPatternNode, DFPatternNode);
};

class ExprPattern : public DFPattern {
 public:
  TVM_DLL explicit ExprPattern(Expr expr);
  TVM_DEFINE_OBJECT_REF_METHODS(ExprPattern, DFPattern, ExprPatternNode);
};

class VarPattern;
class VarPatternNode : public DFPatternNode {
 public:
  String name;
  const String& name_hint() const { return name; }
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("name", &name); }

  static constexpr const char* _type_key = "relax.dataflow_pattern.VarPattern";
  TVM_DECLARE_BASE_OBJECT_INFO(VarPatternNode, DFPatternNode);
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

class DataflowVarPattern : public DFPattern {
 public:
  TVM_DLL DataflowVarPattern(String name_hint);
  TVM_DEFINE_OBJECT_REF_METHODS(DataflowVarPattern, DFPattern, DataflowVarPatternNode);
};

/*!
 * \brief A Pattern to Match a Relax Global Variable
 */
class GlobalVarPattern;
/*! \brief Container for Var */
class GlobalVarPatternNode : public VarPatternNode {
 public:
  static constexpr const char* _type_key = "relax.dataflow_pattern.GlobalVarPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalVarPatternNode, DFPatternNode);
};
class GlobalVarPattern : public DFPattern {
 public:
  TVM_DLL GlobalVarPattern(String name_hint);
  TVM_DEFINE_OBJECT_REF_METHODS(GlobalVarPattern, DFPattern, GlobalVarPatternNode);
};

class ConstantPattern;
class ConstantPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relax.dataflow_pattern.ConstantPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantPatternNode, DFPatternNode);
};

class ConstantPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(ConstantPattern, DFPattern, ConstantPatternNode);
};

class CallPattern;
/*! \brief CallPattern container. */
class CallPatternNode : public DFPatternNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be relay::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  DFPattern op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<DFPattern> args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.CallPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallPatternNode, DFPatternNode);
};

class CallPattern : public DFPattern {
 public:
  TVM_DLL CallPattern(DFPattern op, Array<DFPattern> args);
  TVM_DEFINE_OBJECT_REF_METHODS(CallPattern, DFPattern, CallPatternNode);
};

class PrimArrPattern;
class PrimArrPatternNode : public DFPatternNode {
 public:
  /*! \brief The array to match */
  Array<PrimExpr> array;
  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("array", &array); }
  static constexpr const char* _type_key = "relax.dataflow_pattern.PrimArrPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimArrPatternNode, DFPatternNode);
};
class PrimArrPattern : public DFPattern {
 public:
  TVM_DLL PrimArrPattern(Array<PrimExpr> arr);
  TVM_DEFINE_OBJECT_REF_METHODS(PrimArrPattern, DFPattern, PrimArrPatternNode);
};

/*!
 * \brief Relay Function container
 * \sa Function
 */
class FunctionPatternNode : public DFPatternNode {
 public:
  /*! \brief Function parameters */
  tvm::Array<DFPattern> params;
  /*!
   * \brief
   * The expression which represents the computation of the function,
   * the expression may reference the parameters, and the type of it
   * or sub-expressions may reference the type variables.
   */
  DFPattern body;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("body", &body);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.FunctionPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionPatternNode, DFPatternNode);
};

/*!
 * \brief Managed reference to FunctionNode.
 * \sa FunctionNode
 */
class FunctionPattern : public DFPattern {
 public:
  /*!
   * \brief Constructor
   * \param params The parameters of the function.
   * \param body The body of the function.
   */
  TVM_DLL FunctionPattern(tvm::Array<DFPattern> params, DFPattern body);

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionPattern, DFPattern, FunctionPatternNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionPatternNode);
};

/*! \brief Tuple of multiple Exprs */
class TuplePattern;
/*! \brief Tuple container */
class TuplePatternNode : public DFPatternNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<DFPattern> fields;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("fields", &fields); }

  static constexpr const char* _type_key = "relax.dataflow_pattern.TuplePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TuplePatternNode, DFPatternNode);
};

class TuplePattern : public DFPattern {
 public:
  TVM_DLL explicit TuplePattern(tvm::Array<DFPattern> fields);
  TVM_DEFINE_OBJECT_REF_METHODS(TuplePattern, DFPattern, TuplePatternNode);
};

/*! \brief Get index-th field out of a tuple. */
class TupleGetItemPattern;
class TupleGetItemPatternNode : public DFPatternNode {
 public:
  /*! \brief The tuple Expression */
  DFPattern tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple", &tuple);
    v->Visit("index", &index);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.TupleGetItemPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemPatternNode, DFPatternNode);
};
class TupleGetItemPattern : public DFPattern {
 public:
  TVM_DLL TupleGetItemPattern(DFPattern tuple, int index);
  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItemPattern, DFPattern, TupleGetItemPatternNode);
};

class AndPattern;
class AndPatternNode : public DFPatternNode {
 public:
  DFPattern left, right;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("left", &left);
    v->Visit("right", &right);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.AndPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AndPatternNode, DFPatternNode);
};
class AndPattern : public DFPattern {
 public:
  TVM_DLL AndPattern(DFPattern lhs, DFPattern rhs);
  TVM_DEFINE_OBJECT_REF_METHODS(AndPattern, DFPattern, AndPatternNode);
};

class OrPattern;
/*!
 * \brief Pattern for Alternate Expressions.
 */
class OrPatternNode : public DFPatternNode {
 public:
  /*! \brief The left optional pattern. */
  DFPattern left;
  /*! \brief The right optional pattern. */
  DFPattern right;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("left", &left);
    v->Visit("right", &right);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.OrPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(OrPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches either of two patterns
 */
class OrPattern : public DFPattern {
 public:
  TVM_DLL OrPattern(DFPattern left, DFPattern right);
  TVM_DEFINE_OBJECT_REF_METHODS(OrPattern, DFPattern, OrPatternNode);
};

/*!
 * \brief Wildcard Pattern.
 */
class WildcardPatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relax.dataflow_pattern.WildcardPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(WildcardPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches anything.
 */
class WildcardPattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(WildcardPattern, DFPattern, WildcardPatternNode);
};

class TypePattern;
class TypePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  Type type;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("type", &type);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.TypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(TypePatternNode, DFPatternNode);
};
class TypePattern : public DFPattern {
 public:
  TVM_DLL TypePattern(DFPattern pattern, Type type);
  TVM_DEFINE_OBJECT_REF_METHODS(TypePattern, DFPattern, TypePatternNode);
};

class ShapePattern;
class ShapePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  Array<PrimExpr> shape;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("shape", &shape);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.ShapePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapePatternNode, DFPatternNode);
};
class ShapePattern : public DFPattern {
 public:
  TVM_DLL ShapePattern(DFPattern pattern, Array<PrimExpr> type);
  TVM_DEFINE_OBJECT_REF_METHODS(ShapePattern, DFPattern, ShapePatternNode);
};

class DataTypePattern;
class DataTypePatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The type to match */
  DataType dtype;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("dtype", &dtype);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.DataTypePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DataTypePatternNode, DFPatternNode);
};
class DataTypePattern : public DFPattern {
 public:
  TVM_DLL DataTypePattern(DFPattern pattern, DataType dtype);
  TVM_DEFINE_OBJECT_REF_METHODS(DataTypePattern, DFPattern, DataTypePatternNode);
};

class AttrPattern;
class AttrPatternNode : public DFPatternNode {
 public:
  /*! \brief The pattern. */
  DFPattern pattern;
  /*! \brief The attribute to match */
  DictAttrs attrs;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("pattern", &pattern);
    v->Visit("attrs", &attrs);
  }

  static constexpr const char* _type_key = "relax.dataflow_pattern.AttrPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrPatternNode, DFPatternNode);
};

/*!
 * \brief A pattern which matches attributes in another pattern
 */
class AttrPattern : public DFPattern {
 public:
  TVM_DLL AttrPattern(DFPattern pattern, DictAttrs attrs);
  TVM_DEFINE_OBJECT_REF_METHODS(AttrPattern, DFPattern, AttrPatternNode);
};

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

/*! \brief Syntatic Sugar for creating a VarPattern with a name */
DFPattern IsVar(const String& name);
/*! \brief Syntatic Sugar for creating a ConstantPattern */
DFPattern IsConstant();
/*! \brief Syntatic Sugar for creating a WildcardPattern */
DFPattern IsWildcard();
/*! \brief Syntatic Sugar for creating a ExprPattern */
DFPattern IsExpr(const Expr& expr);
/*! \brief Syntatic Sugar for creating a ExprPattern base on an Op*/
DFPattern IsOp(const String& op_name);
/*! \brief Syntatic Sugar for call_tir (return a tensor) */
DFPattern IsCallTIR(const String& name, const Optional<TuplePattern> args = NullOpt,
                    Optional<Array<PrimExpr>> oshape = NullOpt);
/*! \brief Syntatic Sugar for call_tir (return a tuple of tensor) */
DFPattern IsCallTIR(const String& name, TuplePattern var_args, Array<Array<PrimExpr>> oshapes);
/*! \brief Syntatic Sugar for creating a TuplePattern*/
DFPattern IsTuple(const Array<DFPattern>& fields);
/*! \brief Syntatic Sugar for creating a TupleGetItemPattern*/
DFPattern IsTupleGetItem(const DFPattern tuple, int index = -1);

template <typename... Args>
DFPattern DFPattern::operator()(Args&&... args) const {
  return CallPattern(GetRef<DFPattern>(this->get()),
                     Array<DFPattern>({std::forward<Args>(args)...}));
}

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_PATTERN_H_
