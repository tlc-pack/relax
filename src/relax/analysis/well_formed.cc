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
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr_functor.h>

#include <unordered_set>

namespace tvm {
namespace relax {

/*! \brief Helper to visit PrimExpr in the shape annotation and check if the symbolic vars in the
 * PrimExpr are defined.*/
class PrimExprVisitor : public tir::ExprVisitor {
 public:
  void RegisterSymVar(const tir::Var var) { symbolicvar_set_.insert(var); }
  void ClearSymVarSet() { symbolicvar_set_.clear(); }

 private:
  void VisitExpr_(const tir::VarNode* op) {
    tir::Var var = GetRef<tir::Var>(op);
    if (symbolicvar_set_.count(var) == 0) {
      LOG(FATAL) << "WellFormedCheckError: Symbolic Var " << var->name_hint << " is not defined.";
    }
  }
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> symbolicvar_set_;
};

/*! \brief Helper to implement well formed check.*/
class WellFormedChecker : public relax::ExprVisitor {
 public:
  void RegisterGlobalVar(GlobalVar var) { globalvar_set_.insert(var); }

 private:
  void VisitExpr_(const GlobalVarNode* op) {
    GlobalVar var = GetRef<GlobalVar>(op);
    if (globalvar_set_.count(var) == 0) {
      LOG(FATAL) << "WellFormedCheckError: GlobalVar" << op->name_hint << " is not defined.";
    }
  }

  void VisitExpr_(const TupleNode* op) {
    for (size_t i = 0; i < op->fields.size(); i++) {
      Expr expr = op->fields[i];
      if (expr.as<VarNode>() || expr.as<DataflowVarNode>() || expr.as<ShapeExprNode>() ||
          expr.as<ConstantNode>() || expr.as<TupleNode>()) {
        this->VisitExpr(expr);
      } else {
        LOG(FATAL) << "WellFormedCheckError: Tuple is not in ANF form, field " << i << " gets "
                   << expr->GetTypeKey();
      }
    }

    if (op->shape_) {
      this->VisitExpr(Downcast<Expr>(op->shape_.value()));
    }
  }

  void VisitExpr_(const VarNode* op) {
    Var var = GetRef<Var>(op);
    if (var_set_.count(var) == 0) {
      LOG(FATAL) << "WellFormedCheckError: Var " << op->name_hint() << " is not defined.";
    }
  }

  void VisitExpr_(const DataflowVarNode* op) {
    DataflowVar var = GetRef<DataflowVar>(op);
    if (!is_dataflow) {
      LOG(FATAL) << "WellFormedCheckError: DataflowVar " << op->name_hint()
                 << " is used outside DataflowBlock.";
    }
    if (dataflowvar_set_.count(var) == 0) {
      LOG(FATAL) << "WellFormedCheckError: DataflowVar " << op->name_hint() << " is not defined.";
    }
  }

  void VisitExpr_(const FunctionNode* op) {
    for (Var param : op->params) {
      // register symbolic var defined in the shape annotation of function params
      if (param->shape_) {
        Expr var_shape = Downcast<Expr>(param->shape_);
        if (var_shape.as<RuntimeDepShapeNode>()) {
          VisitExpr(var_shape);
        } else {
          for (PrimExpr expr : Downcast<ShapeExpr>(var_shape)->values) {
            if (expr.as<tir::VarNode>()) {
              prim_expr_visitor.RegisterSymVar(Downcast<tir::Var>(expr));
            } else {
              prim_expr_visitor(expr);
            }
          }
        }
      }

      this->VisitVarDef(param);
    }
    this->VisitBody(op->body);
    var_set_.clear();
    prim_expr_visitor.ClearSymVarSet();
  }

  void VisitExpr_(const CallNode* op) {
    for (size_t i = 0; i < op->args.size(); i++) {
      Expr arg = op->args[i];
      if (arg.as<GlobalVarNode>() || arg.as<ExternFuncNode>() || arg.as<TupleNode>() ||
          arg.as<ShapeExprNode>() || arg.as<VarNode>() || arg.as<DataflowVarNode>() ||
          arg.as<ConstantNode>()) {
        this->VisitExpr(arg);
      } else {
        LOG(FATAL) << "WellFormedCheckError: Call is not in ANF form, arg " << i << " gets "
                   << arg->GetTypeKey();
      }
    }

    if (op->shape_) {
      this->VisitExpr(Downcast<Expr>(op->shape_.value()));
    }
  }

  void VisitExpr_(const IfNode* op) {
    this->VisitExpr(op->cond);
    this->VisitBody(op->true_branch);
    this->VisitBody(op->false_branch);
  }

  void VisitExpr_(const ShapeExprNode* op) {
    for (PrimExpr expr : op->values) {
      // check if the symbolic vars in the expr are defined, e.g, 2 * m
      prim_expr_visitor(expr);
    }
  }

  void VisitExpr_(const SeqExprNode* op) {
    LOG(FATAL) << "WellFormedCheckError: SeqExpr only serves as the function body in FunctionNode, "
                  "or the true/false branch body in IfNode.";
  }

  void VisitSeqExpr(const SeqExprNode* op) {
    // a special call only if SeqExpr is the function body
    // in FunctionNode or the true/false branch body in IfNode
    for (BindingBlock block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    this->VisitExpr(op->body);
  }

  void VisitBody(const Expr& expr) {
    if (const SeqExprNode* seq_expr = expr.as<SeqExprNode>()) {
      this->VisitSeqExpr(seq_expr);
    } else {
      this->VisitExpr(expr);
    }
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
        prim_expr_visitor.RegisterSymVar(Downcast<tir::Var>(expr));
      } else {
        // check if the symbolic var in the expr are defined, e.g, 2 * m
        prim_expr_visitor(expr);
      }
    }

    if (binding->var.defined()) {
      this->VisitVarDef(binding->var);
    }
  }

  void VisitBindingBlock_(const DataflowBlockNode* block) {
    is_dataflow = true;
    for (Binding binding : block->bindings) {
      this->VisitBinding(binding);
    }
    is_dataflow = false;
    dataflowvar_set_.clear();
  }

  void VisitVarDef_(const DataflowVarNode* var) {
    if (!is_dataflow) {
      LOG(FATAL) << "WellFormedCheckError: DataflowVar " << var->name_hint()
                 << " is defined outside DataflowBlock.";
    }
    DataflowVar lv = GetRef<DataflowVar>(var);
    if (dataflowvar_set_.count(lv) == 1) {
      LOG(FATAL) << "WellFormedCheckError: DataflowVar " << lv->name_hint()
                 << " is defined more than once.";
    }
    // register DataflowVar
    dataflowvar_set_.insert(lv);
  }

  void VisitVarDef_(const VarNode* var) {
    Var gv = GetRef<Var>(var);
    if (var_set_.count(gv) == 1) {
      LOG(FATAL) << "WellFormedCheckError: Var " << gv->name_hint()
                 << " is defined more than once.";
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

  bool is_dataflow = false;
  std::unordered_set<GlobalVar, ObjectPtrHash, ObjectPtrEqual> globalvar_set_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> var_set_;
  std::unordered_set<DataflowVar, ObjectPtrHash, ObjectPtrEqual> dataflowvar_set_;
  PrimExprVisitor prim_expr_visitor;
};

void WellFormed(const IRModule& m) {
  WellFormedChecker well_formed_checker = WellFormedChecker();
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
}

TVM_REGISTER_GLOBAL(("relax.analysis.well_formed")).set_body_typed([](IRModule m) {
  return WellFormed(m);
});

}  // namespace relax
}  // namespace tvm