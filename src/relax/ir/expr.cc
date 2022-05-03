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

namespace tvm {

RelayExpr RelayExprNode::shape() const {
  if (this->shape_.defined()) {
    return Downcast<RelayExpr>(this->shape_);
  }
  static const Op& op = Op::Get("relax.shape_of");
  RelayExpr self = GetRef<RelayExpr>(this);
  return relay::Call(op, {self}, {}, {});
}

TVM_REGISTER_GLOBAL("ir.RelayExprShape").set_body_method<RelayExpr>(&RelayExprNode::shape);

namespace relax {
using tvm::ReprPrinter;
using tvm::runtime::Optional;

TVM_REGISTER_NODE_TYPE(ShapeExprNode);

ShapeExpr::ShapeExpr(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeExprNode> n = make_object<ShapeExprNode>();
  n->values = std::move(values);
  n->span = span;
  n->shape_ = NullOpt;
  n->checked_type_ = ShapeType(Span());
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ShapeExpr").set_body_typed([](Array<PrimExpr> values, Span span) {
  return ShapeExpr(values, span);
});

TVM_REGISTER_NODE_TYPE(RuntimeDepShapeNode);

RuntimeDepShape::RuntimeDepShape(Span span) {
  ObjectPtr<RuntimeDepShapeNode> n = make_object<RuntimeDepShapeNode>();
  n->span = span;
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

Function::Function(Array<Var> params, Expr body, Type ret_type, DictAttrs attrs, Span span) {
  // Set the function type.
  // For function, we take a conservative approach and require the function type
  // to be known at construction time.
  Array<Type> param_types;
  for (Var param : params) {
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
  n->checked_type_ = std::move(func_type);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.Function")
    .set_body_typed([](Array<Var> params, Expr body, Type ret_type, DictAttrs attrs, Span span) {
      return Function(params, body, ret_type, attrs, span);
    });

Function Function::CreateUnchecked(Array<Var> params, Expr body, Type ret_type, DictAttrs attrs,
                                   Span span) {
  for (Var param : params) {
    ICHECK(param->checked_type_.defined())
        << "relax.Function requires params to contain checked_type_.";
  }

  // set the fields
  ObjectPtr<FunctionNode> n = make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  return Function(std::move(n));
}

TVM_REGISTER_GLOBAL("relax.Function_CreateUnchecked")
    .set_body_typed([](Array<Var> params, Expr body, Type ret_type, DictAttrs attrs, Span span) {
      return Function::CreateUnchecked(params, body, ret_type, attrs, span);
    });

TVM_REGISTER_NODE_TYPE(ExternFuncNode);

ExternFunc::ExternFunc(String global_symbol, Span span) {
  ObjectPtr<ExternFuncNode> n = make_object<ExternFuncNode>();
  n->global_symbol = std::move(global_symbol);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.ExternFunc").set_body_typed([](String global_symbol, Span span) {
  return ExternFunc(global_symbol, span);
});

void UpdateType(Expr expr, Type type) {
  ICHECK(!expr->checked_type_.defined() || tvm::StructuralEqual()(expr->checked_type_, type))
      << "the checked_type_ of the Expr must not be nullptr for idempotency";
  expr->checked_type_ = type;
}

TVM_REGISTER_GLOBAL("relax.UpdateType").set_body_typed([](Expr expr, Type type) {
  UpdateType(expr, type);
});

void UpdateShape(Expr expr, Optional<ObjectRef> shape) {
  ICHECK(!expr->shape_.defined()) << "the shape_ of the Expr must not be nullptr for idempotency";
  expr->shape_ = shape;
}

TVM_REGISTER_GLOBAL("relax.UpdateShape").set_body_typed([](Expr expr, Optional<ObjectRef> shape) {
  UpdateShape(expr, shape);
});

}  // namespace relax
}  // namespace tvm
