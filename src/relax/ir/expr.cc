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
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_analysis.h>

namespace tvm {

RelayExpr RelayExprNode::shape() const {
  if (this->shape_.defined()) {
    return Downcast<RelayExpr>(this->shape_);
  }
  static const Op& op = Op::Get("relax.shape_of");
  RelayExpr self = GetRef<RelayExpr>(this);
  relax::Call call_shape_of(op, {self}, {}, {});
  call_shape_of->checked_type_ = relax::ShapeType();
  return call_shape_of;
}

TVM_REGISTER_GLOBAL("ir.RelayExprShape").set_body_method<RelayExpr>(&RelayExprNode::shape);

namespace relax {
using tvm::ReprPrinter;
using tvm::runtime::Optional;

Call::Call(Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  data_ = std::move(n);
}

Call WithFields(Call call, Optional<Expr> opt_op, Optional<Array<Expr>> opt_args,
                Optional<Attrs> opt_attrs, Optional<Array<Type>> opt_type_args,
                Optional<Span> opt_span) {
  // Collect new values for fields.
  Expr op = opt_op.value_or(call->op);
  Array<Expr> args = opt_args.value_or(call->args);
  Attrs attrs = opt_attrs.value_or(call->attrs);
  Array<Type> type_args = opt_type_args.value_or(call->type_args);
  Span span = opt_span.value_or(call->span);

  // Check if anything changed.
  bool unchanged = op.same_as(call->op) && attrs.same_as(call->attrs) && span.same_as(call->span);
  if (unchanged) {
    if (args.size() == call->args.size()) {
      for (size_t i = 0; i < args.size(); i++) {
        unchanged &= args[i].same_as(call->args[i]);
      }
    } else {
      unchanged = false;
    }
  }
  if (unchanged) {
    if (type_args.size() == call->type_args.size()) {
      for (size_t i = 0; i < type_args.size(); i++) {
        unchanged &= type_args[i].same_as(call->type_args[i]);
      }
    } else {
      unchanged = false;
    }
  }

  if (!unchanged) {
    // If call is only references, update it in place. Otherwise copy and update.
    CallNode* cow_call_node = call.CopyOnWrite();
    cow_call_node->op = op;
    cow_call_node->args = args;
    cow_call_node->attrs = attrs;
    cow_call_node->type_args = type_args;
    cow_call_node->span = span;
  }
  return call;
}

TVM_REGISTER_NODE_TYPE(CallNode);

TVM_REGISTER_GLOBAL("relax.Call")
    .set_body_typed([](Expr op, Array<Expr> args, Attrs attrs, Array<Type> type_args, Span span) {
      return Call(op, args, attrs, type_args, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<CallNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const CallNode*>(ref.get());
      p->stream << "CallNode(" << node->op << ", " << node->args << ", " << node->attrs << ", "
                << node->type_args << ")";
    });

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}

If WithFields(If if_expr, Optional<Expr> opt_cond, Optional<Expr> opt_true_branch,
              Optional<Expr> opt_false_branch, Optional<Span> opt_span) {
  Expr cond = opt_cond.value_or(if_expr->cond);
  Expr true_branch = opt_true_branch.value_or(if_expr->true_branch);
  Expr false_branch = opt_false_branch.value_or(if_expr->false_branch);
  Span span = opt_span.value_or(if_expr->span);

  bool unchanged = cond.same_as(if_expr->cond) && true_branch.same_as(if_expr->true_branch) &&
                   false_branch.same_as(if_expr->false_branch) && span.same_as(if_expr->span);

  if (!unchanged) {
    IfNode* cow_if_node = if_expr.CopyOnWrite();
    cow_if_node->cond = cond;
    cow_if_node->true_branch = true_branch;
    cow_if_node->false_branch = false_branch;
    cow_if_node->span = span;
  }
  return if_expr;
}

TVM_REGISTER_NODE_TYPE(IfNode);

TVM_REGISTER_GLOBAL("relax.If")
    .set_body_typed([](Expr cond, Expr true_branch, Expr false_branch, Span span) {
      return If(cond, true_branch, false_branch, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const IfNode*>(ref.get());
      p->stream << "IfNode(" << node->cond << ", " << node->true_branch << ", "
                << node->false_branch << ")";
    });

Tuple::Tuple(tvm::Array<relay::Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleNode);

TVM_REGISTER_GLOBAL("relax.Tuple").set_body_typed([](tvm::Array<relay::Expr> fields, Span span) {
  return Tuple(fields, span);
});

Tuple WithFields(Tuple tuple, Optional<Array<Expr>> opt_fields, Optional<Span> opt_span) {
  Array<Expr> fields = opt_fields.value_or(tuple->fields);
  Span span = opt_span.value_or(tuple->span);

  bool all_fields_unchanged = true;
  if (fields.size() == tuple->fields.size()) {
    for (size_t i = 0; i < fields.size(); i++) {
      all_fields_unchanged &= fields[i].same_as(tuple->fields[i]);
    }
  } else {
    all_fields_unchanged = false;
  }

  all_fields_unchanged = all_fields_unchanged && span.same_as(tuple->span);
  if (!all_fields_unchanged) {
    TupleNode* cow_tuple_node = tuple.CopyOnWrite();
    cow_tuple_node->fields = fields;
    cow_tuple_node->span = span;
  }
  return tuple;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleNode*>(ref.get());
      p->stream << "Tuple(" << node->fields << ")";
    });

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem WithFields(TupleGetItem tuple_get_item, Optional<Expr> opt_tuple,
                        Optional<Integer> opt_index, Optional<Span> opt_span) {
  Expr tuple = opt_tuple.value_or(tuple_get_item->tuple);
  Integer index = opt_index.value_or(tuple_get_item->index);
  Span span = opt_span.value_or(tuple_get_item->span);

  bool unchanged = tuple.same_as(tuple_get_item->tuple) && (index == tuple_get_item->index) &&
                   span.same_as(tuple_get_item->span);
  if (!unchanged) {
    TupleGetItemNode* cow_tuple_get_item_node = tuple_get_item.CopyOnWrite();
    cow_tuple_get_item_node->tuple = tuple;
    cow_tuple_get_item_node->index = index.IntValue();
    cow_tuple_get_item_node->span = span;
  }
  return tuple_get_item;
}

TVM_REGISTER_NODE_TYPE(TupleGetItemNode);

TVM_REGISTER_GLOBAL("relax.TupleGetItem").set_body_typed([](Expr tuple, int index) {
  return TupleGetItem(tuple, index);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TupleGetItemNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const TupleGetItemNode*>(ref.get());
      p->stream << "TupleGetItemNode(" << node->tuple << ", " << node->index << ")";
    });

TVM_REGISTER_NODE_TYPE(ShapeExprNode);

ShapeExpr::ShapeExpr(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeExprNode> n = make_object<ShapeExprNode>();
  Array<PrimExpr> new_values;
  new_values.reserve(values.size());
  for (const PrimExpr& value : values) {
    PrimExpr new_value = value;
    if (value->IsInstance<IntImmNode>()) {
      new_value = tvm::cast(DataType::Int(64), value);
    } else if (value.dtype() != DataType::Int(64)) {
      LOG(FATAL) << "the value in ShapeExpr can only have dtype of int64";
    }
    new_values.push_back(new_value);
  }
  n->values = std::move(new_values);
  n->span = span;
  n->shape_ = NullOpt;
  n->checked_type_ = ShapeType();
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ShapeExpr").set_body_typed([](Array<PrimExpr> values, Span span) {
  return ShapeExpr(values, span);
});

TVM_REGISTER_NODE_TYPE(RuntimeDepShapeNode);

RuntimeDepShape::RuntimeDepShape(Span span) {
  ObjectPtr<RuntimeDepShapeNode> n = make_object<RuntimeDepShapeNode>();
  n->span = span;
  n->checked_type_ = ShapeType();
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.RuntimeDepShape").set_body_typed([](Span span) {
  return RuntimeDepShape(span);
});

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ShapeExprNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const ShapeExprNode* node = static_cast<const ShapeExprNode*>(ref.get());
      p->stream << "ShapeExpr(";
      for (auto it = node->values.begin(); it != node->values.end(); it++) {
        if (it != node->values.begin()) {
          p->stream << ", ";
        }
        p->stream << *it;
      }
      p->stream << ")";
    });

TVM_REGISTER_NODE_TYPE(VarNode);

Var::Var(Id vid, Optional<Expr> shape_annotation, Optional<Type> type_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->shape_ = std::move(shape_annotation);
  if (type_annotation) {
    n->checked_type_ = std::move(type_annotation.value());
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Var")
    .set_body_typed([](String name_hint, Optional<Expr> shape_annotation,
                       Optional<Type> type_annotation, Span span) {
      return Var(name_hint, shape_annotation, type_annotation, span);
    });

TVM_REGISTER_GLOBAL("relax.VarFromId")
    .set_body_typed([](Id vid, Optional<Expr> shape_annotation, Optional<Type> type_annotation,
                       Span span) { return Var(vid, shape_annotation, type_annotation, span); });

TVM_REGISTER_NODE_TYPE(DataflowVarNode);

DataflowVar::DataflowVar(Id vid, Optional<Expr> shape_annotation, Optional<Type> type_annotation,
                         Span span) {
  ObjectPtr<DataflowVarNode> n = make_object<DataflowVarNode>();
  n->vid = std::move(vid);
  n->shape_ = std::move(shape_annotation);
  if (type_annotation) {
    n->checked_type_ = std::move(type_annotation.value());
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowVar")
    .set_body_typed([](String name_hint, Optional<Expr> shape_annotation,
                       Optional<Type> type_annotation, Span span) {
      return DataflowVar(name_hint, shape_annotation, type_annotation, span);
    });

TVM_REGISTER_GLOBAL("relax.DataflowVarFromId")
    .set_body_typed([](Id vid, Optional<Expr> shape_annotation, Optional<Type> type_annotation,
                       Span span) {
      return DataflowVar(vid, shape_annotation, type_annotation, span);
    });

Constant::Constant(runtime::NDArray data, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);
  DataType dtype = n->data.DataType();
  ShapeTuple shape_tuple = n->data.Shape();
  Type type = DynTensorType(shape_tuple.size(), dtype);
  n->checked_type_ = type;
  Array<PrimExpr> values;
  for (size_t dim = 0; dim < shape_tuple.size(); dim++) {
    values.push_back(IntImm(DataType::Int(64), shape_tuple[dim]));
  }
  n->shape_ = ShapeExpr(values);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ConstantNode);

TVM_REGISTER_GLOBAL("relax.Constant").set_body_typed([](runtime::NDArray data, Span span = Span()) {
  return Constant(data, span);
});

TVM_REGISTER_NODE_TYPE(BindingNode);

TVM_REGISTER_NODE_TYPE(MatchShapeNode);

MatchShape::MatchShape(Expr value, Array<PrimExpr> pattern, Var var, Span span) {
  ObjectPtr<MatchShapeNode> n = make_object<MatchShapeNode>();
  n->value = std::move(value);
  n->pattern = std::move(pattern);
  n->var = std::move(var);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.MatchShape")
    .set_body_typed([](Expr value, Array<PrimExpr> pattern, Var var, Span span) {
      return MatchShape(value, pattern, var, span);
    });

TVM_REGISTER_NODE_TYPE(VarBindingNode);

VarBinding::VarBinding(Var var, Expr value, Span span) {
  ObjectPtr<VarBindingNode> n = make_object<VarBindingNode>();
  n->var = std::move(var);
  n->value = std::move(value);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.VarBinding").set_body_typed([](Var var, Expr value, Span span) {
  return VarBinding(var, value, span);
});

TVM_REGISTER_NODE_TYPE(BindingBlockNode);

BindingBlock::BindingBlock(Array<Binding> bindings, Span span) {
  ObjectPtr<BindingBlockNode> n = make_object<BindingBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.BindingBlock").set_body_typed([](Array<Binding> bindings, Span span) {
  return BindingBlock(bindings, span);
});

TVM_REGISTER_NODE_TYPE(DataflowBlockNode);

DataflowBlock::DataflowBlock(Array<Binding> bindings, Span span) {
  ObjectPtr<DataflowBlockNode> n = make_object<DataflowBlockNode>();
  n->bindings = std::move(bindings);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowBlock").set_body_typed([](Array<Binding> bindings, Span span) {
  return DataflowBlock(bindings, span);
});

TVM_REGISTER_NODE_TYPE(SeqExprNode);

SeqExpr::SeqExpr(Array<BindingBlock> blocks, Expr body, Span span) {
  ObjectPtr<SeqExprNode> n = make_object<SeqExprNode>();
  n->blocks = std::move(blocks);
  n->body = std::move(body);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.SeqExpr")
    .set_body_typed([](Array<BindingBlock> blocks, Expr body, Span span) {
      return SeqExpr(blocks, body, span);
    });

TVM_REGISTER_NODE_TYPE(FunctionNode);

Function::Function(Array<Var> params, Expr body, Type ret_type, Expr ret_shape, DictAttrs attrs,
                   Span span) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  Array<Type> param_types;
  for (const Var& param : params) {
    CHECK(param->checked_type_.defined())
        << "relax.Function requires params to contain checked_type_";
    param_types.push_back(param->checked_type_);
  }

  if (!ret_type.defined()) {
    CHECK(body->checked_type_.defined())
        << "relax.Function requires body to contain deduced checked_type_"
        << " or ret_type to be supplied";
    ret_type = body->checked_type_;
  } else {
    if (body->checked_type_.defined()) {
      CHECK(IsBaseOf(ret_type, body->checked_type_))
          << "relax.Function requires the deduced body->checked_type_ to be a subtype of the "
             "annotated ret_type but meet body->checked_type_: "
          << body->checked_type_ << ", ret_type: " << ret_type;

      // Use the more refined body->checked_type_ as the return type.
      ret_type = body->checked_type_;
    }
  }
  auto func_type = FuncType(param_types, ret_type, {}, {});

  // set the fields
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->ret_shape = std::move(ret_shape);
  n->checked_type_ = std::move(func_type);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Function")
    .set_body_typed([](Array<Var> params, Expr body, Type ret_type, Expr ret_shape, DictAttrs attrs,
                       Span span) {
      return Function(params, body, ret_type, ret_shape, attrs, span);
    });

Function Function::CreateUnchecked(Array<Var> params, Expr body, Type ret_type, Expr ret_shape,
                                   DictAttrs attrs, Span span) {
  for (Var param : params) {
    ICHECK(param->checked_type_.defined())
        << "relax.Function requires params to contain checked_type_.";
  }

  // set the fields
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->ret_shape = std::move(ret_shape);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_REGISTER_GLOBAL("relax.Function_CreateUnchecked")
    .set_body_typed([](Array<Var> params, Expr body, Type ret_type, Expr ret_shape, DictAttrs attrs,
                       Span span) {
      return Function::CreateUnchecked(params, body, ret_type, ret_shape, attrs, span);
    });

TVM_REGISTER_NODE_TYPE(ExternFuncNode);

ExternFunc::ExternFunc(String global_symbol, Span span) {
  ObjectPtr<ExternFuncNode> n = make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  n->checked_type_ = PackedFuncType();
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ExternFunc").set_body_typed([](String global_symbol, Span span) {
  return ExternFunc(global_symbol, span);
});

void UpdateType(Expr expr, Type type) {
  ICHECK(!expr->checked_type_.defined() || tvm::StructuralEqual()(expr->checked_type_, type))
      << "the checked_type_ of the Expr to be updated must be nullptr for idempotency";
  expr->checked_type_ = type;
}

TVM_REGISTER_GLOBAL("relax.UpdateType").set_body_typed([](Expr expr, Type type) {
  UpdateType(expr, type);
});

void UpdateShape(Expr expr, Optional<ObjectRef> shape) {
  ICHECK(!expr->shape_.defined())
      << "the shape_ of the Expr to be updated must be nullptr for idempotency";
  expr->shape_ = shape;
}

TVM_REGISTER_GLOBAL("relax.UpdateShape").set_body_typed([](Expr expr, Optional<ObjectRef> shape) {
  UpdateShape(expr, shape);
});

}  // namespace relax
}  // namespace tvm
