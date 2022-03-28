/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RELAX_EXPR_H_
#define TVM_RELAX_EXPR_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/span.h>
#include <tvm/node/node.h>
#include <tvm/relax/type.h>
#include <tvm/relay/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace relax {

using Expr = RelayExpr;
using ExprNode = RelayExprNode;
using relay::Call;
using relay::CallNode;
using relay::Constant;
using relay::ConstantNode;
using relay::Id;
using relay::If;
using relay::IfNode;
using relay::Tuple;
using relay::TupleGetItem;
using relay::TupleGetItemNode;
using relay::TupleNode;

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprNode : public ExprNode {
 public:
  /*! The values of the shape expression. */
  Array<PrimExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("shape_", &shape_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ShapeExprNode* other, SEqualReducer equal) const {
    return equal(values, other->values) && equal(checked_type_, other->checked_type_) &&
           equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(values);
    hash_reduce(checked_type_);
    hash_reduce(shape_);
  }

  static constexpr const char* _type_key = "relax.expr.ShapeExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ShapeExprNode, ExprNode);
};

class ShapeExpr : public Expr {
 public:
  TVM_DLL explicit ShapeExpr(Array<PrimExpr> values, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(ShapeExpr, Expr, ShapeExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ShapeExprNode);
};

/*! \brief The variable class for all Relax bindings. */
class VarNode : public ExprNode {
 public:
  /*! \brief The identifier of the variable, which is used for comparing stable equality across
   * transformations. */
  Id vid;
  /*! \brief The type annotation, used by binding sites and parameter declarations. */
  runtime::Optional<Type> type_annotation;

  /*! \return The name hint of the variable */
  const String& name_hint() const { return vid->name_hint; }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("vid", &vid);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
    v->Visit("shape_", &shape_);
  }

  bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(type_annotation, other->type_annotation) &&
           equal(checked_type_, other->checked_type_) && equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
    hash_reduce(type_annotation);
    hash_reduce(shape_);
    hash_reduce(checked_type_);
  }

  static constexpr const char* _type_key = "relax.expr.Var";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 2;
  TVM_DECLARE_BASE_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
  TVM_DLL explicit Var(String name_hint, runtime::Optional<Expr> shape_annotation,
                       runtime::Optional<Type> type_annotation, Span span = Span())
      : Var(Id(name_hint), shape_annotation, type_annotation, span) {}

  TVM_DLL explicit Var(Id vid, runtime::Optional<Expr> shape_annotation,
                       runtime::Optional<Type> type_annotation, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Var, Expr, VarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarNode);
};

/*! \brief A sub-type of the variable node used to mark dataflow variables from
 * normal visible "function local" bindings.
 */
class DataflowVarNode : public VarNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("type_annotation", &type_annotation);
    v->Visit("span", &span);
    v->Visit("shape_", &shape_);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const DataflowVarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(type_annotation, other->type_annotation) &&
           equal(shape_, other->shape_) && equal(checked_type_, other->checked_type_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
    hash_reduce(type_annotation);
    hash_reduce(shape_);
    hash_reduce(checked_type_);
  }

  static constexpr const char* _type_key = "relax.expr.DataflowVar";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowVarNode, VarNode);
};

class DataflowVar : public Var {
 public:
  TVM_DLL explicit DataflowVar(String name_hint, runtime::Optional<Expr> shape_annotation,
                               runtime::Optional<Type> type_annotation, Span span = Span())
      : DataflowVar(Id(name_hint), shape_annotation, type_annotation, span) {}

  TVM_DLL explicit DataflowVar(Id vid, runtime::Optional<Expr> shape_annotation,
                               runtime::Optional<Type> type_annotation, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(DataflowVar, Var, DataflowVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowVarNode);
};

/*! \brief The base class of a variable binding in Relax. */
class BindingNode : public Object {
 public:
  mutable Span span;

  void VisitAttrs(AttrVisitor* v) { v->Visit("span", &span); }
  bool SEqualReduce(const BindingNode* other, SEqualReducer equal) const { return true; }
  void SHashReduce(SHashReducer hash_reduce) const {}

  static constexpr const char* _type_key = "relax.expr.Binding";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(BindingNode, Object);
};

class Binding : public ObjectRef {
 protected:
  Binding() = default;

 public:
  explicit Binding(ObjectPtr<Object> n) : ObjectRef(n) {}
  TVM_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(Binding);
  const BindingNode* operator->() const { return static_cast<const BindingNode*>(data_.get()); }
  const BindingNode* get() const { return operator->(); }
  using ContainerType = BindingNode;
};

/*! \brief Symbolic shape match, binds the variable of the lhs with the rhs. */
class MatchShape;
class MatchShapeNode : public BindingNode {
 public:
  Expr value;
  Array<PrimExpr> pattern;
  Var var;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("value", &value);
    v->Visit("pattern", &pattern);
    v->Visit("var", &var);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const MatchShapeNode* other, SEqualReducer equal) const {
    // NOTE: pattern can contain ShapeExpr which defines the vars
    return equal(value, other->value) && equal.DefEqual(pattern, other->pattern) &&
           equal.DefEqual(var, other->var);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    // NOTE: pattern can contain ShapeExpr which defines the vars
    hash_reduce(value);
    hash_reduce.DefHash(pattern);
    hash_reduce.DefHash(var);
  }

  static constexpr const char* _type_key = "relax.expr.MatchShape";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(MatchShapeNode, BindingNode);
};

class MatchShape : public Binding {
 public:
  TVM_DLL explicit MatchShape(Expr value, Array<PrimExpr> pattern, Var var, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(MatchShape, Binding, MatchShapeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(MatchShapeNode);
};

class VarBinding;
class VarBindingNode : public BindingNode {
 public:
  Var var;
  Expr value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("value", &value);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const VarBindingNode* other, SEqualReducer equal) const {
    return equal.DefEqual(var, other->var) && equal(value, other->value);
  }
  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(var);
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "relax.expr.VarBinding";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(VarBindingNode, BindingNode);
};

class VarBinding : public Binding {
 public:
  TVM_DLL explicit VarBinding(Var var, Expr value, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(VarBinding, Binding, VarBindingNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarBindingNode);
};

class BindingBlock;

class BindingBlockNode : public Object {
 public:
  mutable Span span;
  Array<Binding> bindings;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("span", &span);
    v->Visit("bindings", &bindings);
  }

  bool SEqualReduce(const BindingBlockNode* other, SEqualReducer equal) const {
    return equal(bindings, other->bindings);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(bindings); }

  static constexpr const char* _type_key = "relax.expr.BindingBlock";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_BASE_OBJECT_INFO(BindingBlockNode, Object);
};

class BindingBlock : public ObjectRef {
 public:
  TVM_DLL explicit BindingBlock(Array<Binding> bindings, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(BindingBlock, ObjectRef, BindingBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(BindingBlockNode);
};

class DataflowBlock;
class DataflowBlockNode : public BindingBlockNode {
 public:
  bool SEqualReduce(const DataflowBlockNode* other, SEqualReducer equal) const {
    return equal(bindings, other->bindings);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(bindings); }

  static constexpr const char* _type_key = "relax.expr.DataflowBlock";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(DataflowBlockNode, BindingBlockNode);
};

class DataflowBlock : public BindingBlock {
 public:
  TVM_DLL explicit DataflowBlock(Array<Binding> bindings, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(DataflowBlock, BindingBlock, DataflowBlockNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowBlockNode);
};

/*! \brief A sequence of blocks followed by an expression.
 *
 * The order of blocks enforces scoping and ordering.
 */
class SeqExprNode : public ExprNode {
 public:
  Array<BindingBlock> blocks;
  Expr body;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("blocks", &blocks);
    v->Visit("body", &body);
    v->Visit("shape_", &shape_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const SeqExprNode* other, SEqualReducer equal) const {
    return equal(blocks, other->blocks) && equal(body, other->body) &&
           equal(checked_type_, other->checked_type_) && equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(blocks);
    hash_reduce(body);
    hash_reduce(shape_);
    hash_reduce(checked_type_);
  }

  static constexpr const char* _type_key = "relax.expr.SeqExpr";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(SeqExprNode, ExprNode);
};

class SeqExpr : public Expr {
 public:
  TVM_DLL explicit SeqExpr(Array<BindingBlock> blocks, Expr body, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(SeqExpr, Expr, SeqExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(SeqExprNode);
};

/*! \brief A Relax function, eventually to replace the current Relay function definition. */
class FunctionNode : public BaseFuncNode {
 public:
  /*!
   * \brief Optionally attach the function's name for improved printing, and debugging.
   * It need to be consistent with the GlobalVar in the IRModule.
   */
  runtime::Optional<GlobalVar> name;
  /*! \brief The parameters to the function. */
  Array<Var> params;
  /*! \brief The body of the function. */
  Expr body;
  /*! \brief The return type of the function. */
  Type ret_type;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("name", &name);
    v->Visit("params", &params);
    v->Visit("body", &body);
    v->Visit("ret_type", &ret_type);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("shape_", &shape_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const FunctionNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal.DefEqual(params, other->params) && equal(body, other->body) &&
           equal(ret_type, other->ret_type) && equal(checked_type_, other->checked_type_) &&
           equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(params);
    hash_reduce(body);
    hash_reduce(ret_type);
    hash_reduce(checked_type_);
    hash_reduce(shape_);
  }

  static constexpr const char* _type_key = "relax.expr.Function";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionNode, BaseFuncNode);
};

class Function : public BaseFunc {
 public:
  TVM_DLL explicit Function(runtime::Optional<GlobalVar> name, Array<Var> params, Expr body,
                            Type ret_type, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Function, BaseFunc, FunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
};

/*! \brief The extern function, which can represent packed function. */
class ExternFuncNode : public BaseFuncNode {
 public:
  /*! \brief The name of global symbol. */
  String global_symbol;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("global_symbol", &global_symbol);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const ExternFuncNode* other, SEqualReducer equal) const {
    return equal(global_symbol, other->global_symbol);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(global_symbol); }

  static constexpr const char* _type_key = "relax.expr.ExternFunc";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(ExternFuncNode, BaseFuncNode);
};

class ExternFunc : public BaseFunc {
 public:
  TVM_DLL ExternFunc(String global_symbol, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(ExternFunc, BaseFunc, ExternFuncNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ExternFuncNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_EXPR_H_
