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
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

using Expr = RelayExpr;
using ExprNode = RelayExprNode;
using relay::Id;

/*!
 * \brief Base type of all structure information.
 *
 * StructInfo stores possible structure information
 * deduced during compile-time. It encapsulates
 * both static type and runtime information such
 * as shape.
 *
 * StructInfo of each non-primitive Expr can be
 * deduced during compilation in a "best-effort" manner.
 *
 * When struct_info appears in function parameter and return
 * signatures. They will imply a runtime check that matches
 * the structure information with the value.
 *
 * When it appears in Expr, they follow "assume-semantics",
 * which means the compiler will take the deduced information as it is
 * and only do best effort prove and checks.
 *
 * Each struct info can be uniquely erased to a static-type.
 * The compiler will still compile the code(with less information)
 * when we erase to the static type.
 *
 * If an StructInfo contains an Expr field, then that field
 * must be normalized already through NormalizeArg.
 * This invariant will be checked in constructors
 * and help us to simplify our assumption
 * during struct info deduction.
 */
class StructInfoNode : public Object {
 public:
  /*!
   * \brief Span that points to the original source code.
   *        Reserved debug information.
   */
  mutable Span span;

  static constexpr const char* _type_key = "StructInfo";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  static constexpr const uint32_t _type_child_slots = 5;
  TVM_DECLARE_BASE_OBJECT_INFO(StructInfoNode, Object);
};

/*!
 * \brief Managed reference to StructInfoNode.
 * \sa StructInfoNode
 */
class StructInfo : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(StructInfo, ObjectRef, StructInfoNode);
};

/*!
 * \brief Call corresponds to callable invocation.
 *  Corresponds to operation in computational graph terminology.
 */
class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

  /*!
   * \brief The type arguments passed to polymorphic(template) function.
   *
   * This is the advance feature that is only used when the function is
   * polymorphic. It is safe to be ignored in most cases. For example, in the
   * following code, the type_args of addone call is [int].
   *
   * \code
   *
   * template<typename T>
   * T addone(T a) { return a + 1; }
   *
   * void main() {
   *   int x = addone<int>(10);
   * }
   *
   * \endcode
   */
  tvm::Array<Type> type_args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("op", &op);
    v->Visit("args", &args);
    v->Visit("attrs", &attrs);
    v->Visit("type_args", &type_args);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
  }

  bool SEqualReduce(const CallNode* other, SEqualReducer equal) const {
    // skip type_args check for primitive ops.
    equal->MarkGraphNode();
    return equal(op, other->op) && equal(args, other->args) && equal(attrs, other->attrs) &&
           (IsPrimitiveOp(op) || equal(type_args, other->type_args));
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(op);
    hash_reduce(args);
    hash_reduce(attrs);
    if (!IsPrimitiveOp(op)) {
      hash_reduce(type_args);
    }
  }

  static constexpr const char* _type_key = "relax.expr.Call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
};

class Call : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param op The operator to be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param type_args The type arguments passed to a polymorphic function.
   * \param span The source span of the expression.
   */
  TVM_DLL Call(Expr op, Array<Expr> args, Attrs attrs = Attrs(),
               Array<Type> type_args = Array<Type>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Call, Expr, CallNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(CallNode);
};

/*!
 * \brief Returns \p call with the given properties. A null property denotes 'no change'.
 * Returns \p call if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Call WithFields(Call call, Optional<Expr> opt_op = Optional<Expr>(),
                Optional<Array<Expr>> opt_args = Optional<Array<Expr>>(),
                Optional<Attrs> opt_attrs = Optional<Attrs>(),
                Optional<Array<Type>> opt_type_args = Optional<Array<Type>>(),
                Optional<Span> opt_span = Optional<Span>());

/*!
 * \brief Condition expression
 *
 * Unlike traditional statement `if`s, the if evalutes
 * to the result of the branch taken.
 *
 * x = if (true) { 1 } else { 0 }; // x is 1
 * y = if (false) { 1 } else { 0 }; // y is 0
 *
 * \note This is similar to C's ternary operator.
 */
class IfNode : public ExprNode {
 public:
  /*! \brief The condition. */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  Expr true_branch;
  /*! \brief The expression evaluated when condition is false */
  Expr false_branch;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("cond", &cond);
    v->Visit("true_branch", &true_branch);
    v->Visit("false_branch", &false_branch);
    v->Visit("span", &span);
    v->Visit("shape_", &shape_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("struct_info_", &struct_info_);
  }

  bool SEqualReduce(const IfNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(cond, other->cond) && equal(true_branch, other->true_branch) &&
           equal(false_branch, other->false_branch);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce(cond);
    hash_reduce(true_branch);
    hash_reduce(false_branch);
  }

  static constexpr const char* _type_key = "relax.expr.If";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfNode, ExprNode);
};

class If : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param cond The condition of a if node.
   * \param true_branch The fall through branch
   * \param false_branch The branch for execution when condition is false.
   * \param span The source span of the expression.
   */
  TVM_DLL If(Expr cond, Expr true_branch, Expr false_branch, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(If, Expr, IfNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IfNode);
};

/*!
 * \brief Returns \p if_expr with the given properties. A null property denotes 'no change'.
 * Returns \p if_expr if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
If WithFields(If if_expr, Optional<Expr> opt_cond = Optional<Expr>(),
              Optional<Expr> opt_true_branch = Optional<Expr>(),
              Optional<Expr> opt_false_branch = Optional<Expr>(),
              Optional<Span> opt_span = Optional<Span>());

/*! \brief Tuple container */
class TupleNode : public ExprNode {
 public:
  /*! \brief the fields of the tuple */
  tvm::Array<Expr> fields;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("fields", &fields);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
  }

  bool SEqualReduce(const TupleNode* other, SEqualReducer equal) const {
    // specially handle empty tuple as a constant is not a graph node.
    if (fields.size() == other->fields.size() && fields.size() == 0) {
      return true;
    } else {
      equal->MarkGraphNode();
      return equal(fields, other->fields);
    }
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    if (fields.size() != 0) {
      hash_reduce->MarkGraphNode();
      hash_reduce(fields);
    }
  }

  static constexpr const char* _type_key = "relax.expr.Tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class Tuple : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param fields The fields of a tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Tuple(tvm::Array<Expr> fields, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, Expr, TupleNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleNode);
};

/*!
 * \brief Returns \p tuple with the given properties. A null property denotes 'no change'.
 * Returns \p tuple if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields = Optional<Array<Expr>>(),
                 Optional<Span> opt_span = Optional<Span>());

/*! \brief Get index-th field out of a tuple. */
class TupleGetItemNode : public ExprNode {
 public:
  /*! \brief The tuple Expression */
  Expr tuple;
  /*! \brief which value to get */
  int index;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("tuple_value", &tuple);
    v->Visit("index", &index);
    v->Visit("span", &span);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
  }

  bool SEqualReduce(const TupleGetItemNode* other, SEqualReducer equal) const {
    return equal(tuple, other->tuple) && equal(index, other->index);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(tuple);
    hash_reduce(index);
  }

  static constexpr const char* _type_key = "relax.expr.TupleGetItem";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemNode, ExprNode);
};

class TupleGetItem : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param tuple The tuple to get an element from.
   * \param index The index for extracting a value in the tuple.
   * \param span The source span of the expression.
   */
  TVM_DLL TupleGetItem(Expr tuple, int index, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, Expr, TupleGetItemNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(TupleGetItemNode);
};

/*!
 * \brief Returns \p tuple_get_item with the given properties. A null property denotes 'no change'.
 * Returns \p tuple_get_item if all properties are unchanged. Otherwise, returns a copy with the new
 * fields.
 */
TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple = Optional<Expr>(),
                        Optional<Integer> opt_index = Optional<Integer>(),
                        Optional<Span> opt_span = Optional<Span>());

/*! \brief A shape expression which allows users to construct a shape containing PrimExpr.
 */
class ShapeExprNode : public ExprNode {
 public:
  /*! The values of the shape expression. */
  Array<PrimExpr> values;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("values", &values);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
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

/*! \brief Runtime dependent shape expression.
 *
 * Sometimes shape of a tensor cannot be deduced statically either because the shape is truly data
 * dependent such as output of `unique` operator or cannot be deduced because of limited shape
 * inference capability.
 */
class RuntimeDepShapeNode : public ExprNode {
 public:
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const RuntimeDepShapeNode* other, SEqualReducer equal) const {
    return equal(checked_type_, other->checked_type_) && equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(checked_type_);
    hash_reduce(shape_);
  }

  static constexpr const char* _type_key = "relax.expr.RuntimeDepShape";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(RuntimeDepShapeNode, ExprNode);
};

class RuntimeDepShape : public Expr {
 public:
  TVM_DLL explicit RuntimeDepShape(Span span = Span());
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(RuntimeDepShape, Expr, RuntimeDepShapeNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(RuntimeDepShapeNode);
};

/*! \brief The variable class for all Relax bindings. */
class VarNode : public ExprNode {
 public:
  /*! \brief The identifier of the variable, which is used for comparing stable equality across
   * transformations. */
  Id vid;

  /*! \return The name hint of the variable */
  const String& name_hint() const { return vid->name_hint; }

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("vid", &vid);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const VarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(checked_type_, other->checked_type_) &&
           equal(shape_, other->shape_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
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
  TVM_DLL explicit Var(String name_hint, Optional<StructInfo> struct_info_annotation,
                       Span span = Span())
      : Var(Id(name_hint), struct_info_annotation, span) {}

  TVM_DLL explicit Var(Id vid, Optional<StructInfo> struct_info_annotation, Span span = Span());
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
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const DataflowVarNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal(vid, other->vid) && equal(shape_, other->shape_) &&
           equal(checked_type_, other->checked_type_);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vid);
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
  TVM_DLL explicit DataflowVar(String name_hint, Optional<StructInfo> struct_info_annotation,
                               Span span = Span())
      : DataflowVar(Id(name_hint), struct_info_annotation, span) {}

  TVM_DLL explicit DataflowVar(Id vid, Optional<StructInfo> struct_info_annotation,
                               Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(DataflowVar, Var, DataflowVarNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(DataflowVarNode);
};

/*!
 * \brief Constant tensor.
 *
 * \note Scalar constants are represented by ndim-0 constant tensors.
 */
class ConstantNode : public ExprNode {
 public:
  /*! \brief The data of the tensor */
  runtime::NDArray data;

  /*! \return The corresponding tensor type of the data */
  TensorType tensor_type() const;

  /*! \return Whether it is scalar(ndim-0 tensor) */
  bool is_scalar() const { return data->ndim == 0; }

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("data", &data);
    v->Visit("span", &span);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
  }

  bool SEqualReduce(const ConstantNode* other, SEqualReducer equal) const {
    return equal(data, other->data);
  }

  void SHashReduce(SHashReducer hash_reduce) const { hash_reduce(data); }

  static constexpr const char* _type_key = "relax.expr.Constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit Constant(runtime::NDArray data, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Constant, Expr, ConstantNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(ConstantNode);
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
    v->Visit("struct_info_", &struct_info_);
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

/*! \brief A Relax function. */
class FunctionNode : public BaseFuncNode {
 public:
  /*! \brief The parameters to the function. */
  Array<Var> params;
  /*! \brief The body of the function. */
  Expr body;
  /*! \brief The return type of the function. */
  StructInfo ret_struct_info;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("params", &params);
    v->Visit("body", &body);
    v->Visit("ret_struct_info", &ret_struct_info);
    v->Visit("_checked_type_", &checked_type_);
    v->Visit("shape_", &shape_);
    v->Visit("struct_info_", &struct_info_);
    v->Visit("attrs", &attrs);
    v->Visit("span", &span);
  }

  bool SEqualReduce(const FunctionNode* other, SEqualReducer equal) const {
    equal->MarkGraphNode();
    return equal.DefEqual(params, other->params) && equal(body, other->body) &&
           equal(ret_struct_info, other->ret_struct_info) &&
           equal(checked_type_, other->checked_type_) && equal(shape_, other->shape_) &&
           equal(attrs, other->attrs);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce->MarkGraphNode();
    hash_reduce.DefHash(params);
    hash_reduce(body);
    hash_reduce(ret_struct_info);
    hash_reduce(checked_type_);
    hash_reduce(shape_);
    hash_reduce(attrs);
  }

  static constexpr const char* _type_key = "relax.expr.Function";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionNode, BaseFuncNode);
};

class Function : public BaseFunc {
 public:
  TVM_DLL explicit Function(Array<Var> params, Expr body, Optional<StructInfo> ret_struct_info,
                            DictAttrs attrs = NullValue<DictAttrs>(), Span span = Span());

  /*!
   * \brief Mimics the constructor but without body Expr.
   * \note ret_struct_info is required, since it can not deduced by the body
   */
  TVM_DLL static Function CreateEmpty(Array<Var> params, StructInfo ret_struct_info,
                                      DictAttrs attrs = NullValue<DictAttrs>(), Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(Function, BaseFunc, FunctionNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionNode);
};

// TODO(@sunggg): Investigate the exact usage of kComposite, kPartitionedFromPattern, and
// kPrimitive.
namespace attr {
/*! \brief Mark the function as a primitive function. */
constexpr const char* kPrimitive = "Primitive";
/*!
 * \brief Indicate the codegen that should be used for building this function.
 * When this is unset or set to "default", the default compilation pipeline will be used.
 */
constexpr const char* kCodegen = "Codegen";
/*! \brief Treat the function as a composite operator. */
constexpr const char* kComposite = "Composite";
/*! \brief Indicate the function was created by the Pattern Partitioning Pass. */
constexpr const char* kPartitionedFromPattern = "PartitionedFromPattern";
}  // namespace attr

/*! \brief The extern function, which can represent packed function. */
class ExternFuncNode : public BaseFuncNode {
 public:
  /*! \brief The name of global symbol. */
  String global_symbol;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("global_symbol", &global_symbol);
    v->Visit("struct_info_", &struct_info_);
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
