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
 * \file relax/analysis/well_formed.cc
 * \brief Check if the IRModule is well-formed.
 * This pass is supposed to be applied to normalized Relax AST.
 * If it's malformed, messages will be logged as Warning.
 * This pass will check:
 *    1. GlobalVars are defined before use.
 *    2. Vars are defined before use.
 *    3. Vars are defined exactly once.
 *    4. Symbolic Vars are defined before use.
 *    5. DataflowVars cannot be defined inside BindingBlock.
 *    6. Vars defined in IfNode, except the return Var, are invisible
 *       out of the If body.(May change for new AST designs)
 *    6. SeqExpr only serves as function body, or in the true and
 *       false branches in IfNode.
 *    7. The IR is in ANF:
 *       (a) Expressions cannot contain nested complex expressions.
 *           Here are the expressions that may be nested inside other expressions:
 *           Var, DataflowVar, GlobalVar, Constant, ShapeExpr, RuntimeDepShape,
 *           Op, Tuple (we call these "leaf" expressions).
 *       (b) The right-hand side of a binding may contain a non-leaf expression 
 *           (where all expressions nested in it are leaf expressions),
 *           other than SeqExprs (see rule 6)
 *       (c) Exceptions: The body of a Function node and the true branch
 *           and false branch of If nodes *must* be SeqExprs.
 *       (d) Places where non-leaf expressions cannot appear:
 *           * The tuple_value field of TupleGetItem nodes
 *           * The cond field of If nodes
 *           * The op or args fields of Call nodes
 *           * Inside the fields of Tuple nodes
 *    8. Expr always has checked_type_ (with the exception of Op).
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_set>

namespace tvm {
namespace relax {

class WellFormedChecker;

/*! \brief Helper to visit PrimExpr in the shape annotation and check if the symbolic vars in
 * the PrimExpr are defined.*/
class PrimExprVisitor : public tir::ExprVisitor {
 public:
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> symbolic_var_set_;
  WellFormedChecker* checker_;

  explicit PrimExprVisitor(WellFormedChecker* checker) : checker_(checker) {}

  void VisitExpr_(const tir::VarNode* op);
};

/*! \brief Helper to implement well formed check.*/
class WellFormedChecker : public relax::ExprVisitor {
 public:
  Optional<DiagnosticContext> diag_ctx;

  bool well_formed = true;

  explicit WellFormedChecker(const Optional<DiagnosticContext>& ctx)
      : diag_ctx(ctx), prim_expr_visitor_(this) {}

  void Malformed(Diagnostic diag) {
    well_formed = false;
    LOG(WARNING) << "This IR is not well formed: " << diag->message;
  }

  void VisitExpr(const Expr& expr) override {
    if (!expr->checked_type_.defined()) {
      Malformed(Diagnostic::Error(expr->span)
                << "The checked_type_ of Expr " << expr << " is nullptr.");
    }
    ExprVisitor::VisitExpr(expr);
  }

  void RegisterGlobalVar(GlobalVar var) { global_var_set_.insert(var); }

 private:
  void VisitExpr_(const GlobalVarNode* op) {
    GlobalVar var = GetRef<GlobalVar>(op);
    if (global_var_set_.count(var) == 0) {
      Malformed(Diagnostic::Error(var->span)
                << "GlobalVar " << op->name_hint << " is not defined.");
    }

    if (op->checked_type_.defined()) {
      if ((!op->checked_type_->IsInstance<FuncTypeNode>()) &&
          (!op->checked_type_->IsInstance<PackedFuncTypeNode>())) {
        Malformed(Diagnostic::Error(var->span) << "The checked_type_ of GlobalVar " << op->name_hint
                                               << " must be either FuncType or PackedFuncType.");
      }
    }
  }

  void VisitExpr_(const TupleNode* op) {
    for (size_t i = 0; i < op->fields.size(); i++) {
      Expr expr = op->fields[i];
      if (IsLeafExpr(expr)) {
        this->VisitExpr(expr);
      } else {
        Malformed(Diagnostic::Error(expr->span)
                  << "Tuple is not in ANF form, field " << i << " gets " << expr->GetTypeKey());
      }
    }

    if (op->shape_) {
      this->VisitExpr(Downcast<Expr>(op->shape_.value()));
    }
  }

  void VisitExpr_(const TupleGetItemNode* op) {
    if (IsLeafExpr(op->tuple)) {
      this->VisitExpr(op->tuple);
    } else {
      Malformed(Diagnostic::Error(op->span)
                << "The tuple value in a TupleGetItem node must be a leaf expression.");
    }
  }

  void VisitExpr_(const VarNode* op) {
    Var var = GetRef<Var>(op);
    if (var_set_.count(var) == 0) {
      Malformed(Diagnostic::Error(var->span) << "Var " << op->name_hint() << " is not defined.");
    }
  }

  void VisitExpr_(const DataflowVarNode* op) {
    DataflowVar var = GetRef<DataflowVar>(op);
    if (!is_dataflow_) {
      Malformed(Diagnostic::Error(var->span)
                << "DataflowVar " << op->name_hint() << " is used outside DataflowBlock.");
    }
    if (dataflow_var_set_.count(var) == 0) {
      Malformed(Diagnostic::Error(var->span)
                << "DataflowVar " << op->name_hint() << " is not defined.");
    }
  }

  void VisitExpr_(const FunctionNode* op) {
    // save the var_set_ for local function
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> previous_var_set_ = var_set_;
    for (Var param : op->params) {
      // register symbolic var defined in the shape annotation of function params
      if (param->shape_) {
        Expr var_shape = Downcast<Expr>(param->shape_);
        if (var_shape.as<RuntimeDepShapeNode>() || var_shape.as<TupleNode>()) {
          VisitExpr(var_shape);
        } else {
          for (PrimExpr expr : Downcast<ShapeExpr>(var_shape)->values) {
            if (expr.as<tir::VarNode>()) {
              prim_expr_visitor_.symbolic_var_set_.insert(Downcast<tir::Var>(expr));
            } else {
              prim_expr_visitor_(expr);
            }
          }
        }
      }

      this->VisitVarDef(param);
    }
    if (auto seq = op->body.as<SeqExprNode>()) {
      this->VisitSeqExpr(seq);
    } else {
      Malformed(Diagnostic::Error(op->span) << "Function bodies must be sequence expressions");
    }
    var_set_ = previous_var_set_;
    prim_expr_visitor_.symbolic_var_set_.clear();
  }

  void VisitExpr_(const CallNode* op) {
    for (size_t i = 0; i < op->args.size(); i++) {
      Expr arg = op->args[i];
      if (IsLeafExpr(arg)) {
        this->VisitExpr(arg);
      } else {
        Malformed(Diagnostic::Error(arg->span)
                  << "Call is not in ANF form, arg " << i << " gets " << arg->GetTypeKey());
      }
    }

    if (op->shape_) {
      this->VisitExpr(Downcast<Expr>(op->shape_.value()));
    }
  }

  void VisitExpr_(const IfNode* op) {
    if (IsLeafExpr(op->cond)) {
      this->VisitExpr(op->cond);
    } else {
      Malformed(Diagnostic::Error(op->span)
                << "The condition for an if node must be a leaf expression.");
    }
    auto true_seq = op->true_branch.as<SeqExprNode>();
    auto false_seq = op->false_branch.as<SeqExprNode>();
    if (true_seq && false_seq) {
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> previous_var_set_ = var_set_;
      std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> previous_symbolic_var_set_ =
          prim_expr_visitor_.symbolic_var_set_;
      this->VisitSeqExpr(true_seq);
      var_set_ = previous_var_set_;
      prim_expr_visitor_.symbolic_var_set_ = previous_symbolic_var_set_;
      this->VisitSeqExpr(false_seq);
      var_set_ = previous_var_set_;
      prim_expr_visitor_.symbolic_var_set_ = previous_symbolic_var_set_;
    } else {
      Malformed(Diagnostic::Error(op->span) << "If node branches must be seq exprs");
    }
  }

  void VisitExpr_(const ShapeExprNode* op) {
    for (PrimExpr expr : op->values) {
      // check if the symbolic vars in the expr are defined, e.g, 2 * m
      prim_expr_visitor_(expr);
      if (!expr.dtype().is_int()) {
        Malformed(Diagnostic::Error(expr->span)
                  << "Shape expressions must be of integer type, but got " << expr.dtype());
      }
    }
  }

  void VisitExpr_(const SeqExprNode* op) {
    Malformed(Diagnostic::Error(op->span)
              << "SeqExpr only serves as the function body in FunctionNode, "
                 "or the true/false branch body in IfNode.");
  }

  void VisitSeqExpr(const SeqExprNode* op) {
    // a special call only if SeqExpr is the function body
    // in FunctionNode or the true/false branch body in IfNode
    for (BindingBlock block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    if (!IsLeafExpr(op->body)) {
      Malformed(Diagnostic::Error(op->span) << "SeqExpr bodies must be leaf expressions.");
    }
    this->VisitExpr(op->body);
  }

  void VisitBinding_(const VarBindingNode* binding) {
    this->VisitExpr(binding->value);
    this->VisitVarDef(binding->var);
  }

  void VisitBinding_(const MatchShapeNode* binding) {
    this->VisitExpr(binding->value);
    for (PrimExpr expr : binding->pattern) {
      if (expr.as<tir::VarNode>()) {
        // register symbolic var implicitly defined in the pattern of MatchShape
        prim_expr_visitor_.symbolic_var_set_.insert(Downcast<tir::Var>(expr));
      } else {
        // check if the symbolic var in the expr are defined, e.g, 2 * m
        prim_expr_visitor_(expr);
      }
    }

    if (binding->var.defined()) {
      this->VisitVarDef(binding->var);
    }
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    is_dataflow_ = true;
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    is_dataflow_ = false;
    dataflow_var_set_.clear();
  }

  void VisitVarDef_(const DataflowVarNode* var) {
    if (!is_dataflow_) {
      Malformed(Diagnostic::Error(var->span)
                << "DataflowVar " << var->name_hint() << " is defined outside DataflowBlock.");
    }
    DataflowVar lv = GetRef<DataflowVar>(var);
    if (dataflow_var_set_.count(lv) == 1) {
      Malformed(Diagnostic::Error(var->span)
                << "DataflowVar " << lv->name_hint() << " is defined more than once.");
    }
    // register DataflowVar
    dataflow_var_set_.insert(lv);
  }

  void VisitVarDef_(const VarNode* var) {
    Var gv = GetRef<Var>(var);
    if (var_set_.count(gv) == 1) {
      Malformed(Diagnostic::Error(var->span)
                << "Var " << gv->name_hint() << " is defined more than once.");
    }
    // register Var
    var_set_.insert(gv);
  }

  void VisitVarDef(const Var& var) {
    if (const DataflowVarNode* lv_node = var.as<DataflowVarNode>()) {
      VisitVarDef_(lv_node);
    } else if (const VarNode* gv_node = var.as<VarNode>()) {
      VisitVarDef_(gv_node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << var->GetTypeKey();
    }

    if (var->shape_) {
      VisitExpr(Downcast<Expr>(var->shape_.value()));
    }
  }

  bool is_dataflow_ = false;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> global_var_set_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set_;
  std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> dataflow_var_set_;
  PrimExprVisitor prim_expr_visitor_;
};

void PrimExprVisitor::VisitExpr_(const tir::VarNode* op) {
  tir::Var var = GetRef<tir::Var>(op);
  if (symbolic_var_set_.count(var) == 0) {
    checker_->Malformed(Diagnostic::Error(var->span)
                        << "Symbolic Var " << var->name_hint << " is not defined.");
  }
}

bool WellFormed(const IRModule& m, Optional<DiagnosticContext> diag_ctx) {
  WellFormedChecker well_formed_checker = WellFormedChecker(diag_ctx);
  for (const auto& it : m->functions) {
    // register GlobalVar in the IRModule first
    well_formed_checker.RegisterGlobalVar(it.first);
  }

  for (const auto& it : m->functions) {
    // visit relax.Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      well_formed_checker.VisitExpr(func);
    }
  }

  return well_formed_checker.well_formed;
}

TVM_REGISTER_GLOBAL(("relax.analysis.well_formed")).set_body_typed([](IRModule m) {
  return WellFormed(m);
});

}  // namespace relax
}  // namespace tvm
