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
 * \file src/relax/ir/stmt_rewrite.cc
 * \brief Implementation of statement rewriters.
 */

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/stmt_rewrite.h>

namespace tvm {
namespace relax {

String DataflowBlockRewriteNode::make_new_varname() {
  while (true) {
    String name = "tmp" + std::to_string(++counter_);
    if (used_names_.cend() == used_names_.find(name)) return name;
  }
}

TVM_REGISTER_NODE_TYPE(DataflowBlockRewriteNode);
DataflowBlockRewrite::DataflowBlockRewrite(DataflowBlock dfb, Function root_fn) {
  auto n = make_object<DataflowBlockRewriteNode>();
  n->dfb_ = dfb;
  n->root_fn_ = root_fn;
  n->original_fn_ptr_ = root_fn.get();
  n->to_users_ = FunctionUseDef(root_fn);
  for (const auto& kv : n->to_users_) n->used_names_.insert(kv.first->name_hint());

  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relax.DataflowBlockRewrite")
    .set_body_typed([](DataflowBlock dfb, Function root_fn) {
      return DataflowBlockRewrite(dfb, root_fn);
    });

void DataflowBlockRewriteNode::ReplaceAllUses(Var old_var, Var new_var) {
  class ReplaceAllUsePass : public ExprMutator {
    Var old_var, new_var;
    const DataflowBlockNode* const to_catch;

   public:
    const DataflowBlockNode* caught = nullptr;

    ReplaceAllUsePass(Var old_var, Var new_var, const DataflowBlockNode* to_catch)
        : old_var(old_var), new_var(new_var), to_catch(to_catch) {}

    using ExprMutator::VisitExpr_;

    Expr VisitExpr_(const VarNode* op) override {
      return (op == old_var.get()) ? new_var : GetRef<Expr>(op);
    }

    Expr VisitExpr_(const DataflowVarNode* op) override {
      return (op == old_var.get()) ? new_var : GetRef<Expr>(op);
    }

    BindingBlock VisitBindingBlock_(const DataflowBlockNode* op) override {
      BindingBlock res = ExprMutator::VisitBindingBlock_(op);
      if (op == to_catch) caught = static_cast<const DataflowBlockNode*>(res.get());
      return res;
    }
  };

  ICHECK(to_users_.find(old_var) != to_users_.end()) << "Cannot find " << old_var;
  ICHECK(to_users_.find(new_var) != to_users_.end()) << "Cannot find " << new_var;

  // replace uses in side the DataflowBlock.
  ReplaceAllUsePass replacer(old_var, new_var, dfb_.get());
  root_fn_ = Downcast<Function>(replacer.VisitExpr_(root_fn_.get()));
  dfb_ = GetRef<DataflowBlock>(replacer.caught);

  // update udchain
  // old_var -> old_var users | changed to {}
  // new_var -> {?}           | changed to old_var users
  for (Var user : to_users_[old_var]) {
    Array<Var> new_var_uses = to_users_.Get(new_var).value();
    if (new_var_uses.end() == std::find(new_var_uses.begin(), new_var_uses.end(), user)) {
      new_var_uses.push_back(user);
    }
  }

  to_users_.Set(old_var, {});
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_replace_all_uses")
    .set_body_typed([](DataflowBlockRewrite rwt, Var old_var, Var new_var) {
      rwt->ReplaceAllUses(old_var, new_var);
    });

class UpdateDFB : public ExprMutator {
 private:
  const DataflowBlockNode* const old_dfb_ptr;
  DataflowBlock new_dfb;

 public:
  UpdateDFB(const DataflowBlockNode* to_catch, DataflowBlock new_dfb)
      : old_dfb_ptr(to_catch), new_dfb(std::move(new_dfb)) {}

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* op) override {
    return old_dfb_ptr == op ? new_dfb : GetRef<DataflowBlock>(op);
  }
};

void DataflowBlockRewriteNode::Add(Binding binding) {
  auto p = [binding] {
    if (auto vb = binding.as<VarBindingNode>()) {
      return std::make_pair(vb->var, vb->value);
    } else if (auto ms = binding.as<MatchShapeNode>()) {
      return std::make_pair(ms->var, ms->value);
    }
    LOG(FATAL) << "Unsupported binding type";
    return std::make_pair(Var{}, Expr{});
  }();

  Var var = p.first;
  Expr val = p.second;

  ICHECK(0 == to_users_.count(var)) << var << " has been defined so cannot be added.";

  // Add this VarBinding statement after the definition of uses.
  std::set<const VarNode*> used_vars = [val] {
    class UsedVars : public ExprVisitor {
     public:
      std::set<const VarNode*> used_vars;
      void VisitExpr_(const VarNode* op) override { used_vars.insert(op); }
      void VisitExpr_(const DataflowVarNode* op) override { used_vars.insert(op); }
    } uvar{};
    uvar.VisitExpr(val);
    return std::move(uvar.used_vars);
  }();

  size_t line_last_req_def = 0;
  for (size_t i = 0; i < dfb_->bindings.size(); ++i) {
    auto line = dfb_->bindings[i];
    if (auto varbind = line.as<VarBindingNode>()) {
      if (used_vars.find(varbind->var.get()) != used_vars.cend()) line_last_req_def = i;
    } else if (auto mshape = line.as<MatchShapeNode>()) {
      if (used_vars.find(mshape->var.get()) != used_vars.cend()) line_last_req_def = i;
    }
  }

  auto prev_dfb_ptr = dfb_.get();

  dfb_.object.CopyOnWrite()->bindings.insert(dfb_->bindings.begin() + 1 + line_last_req_def,
                                             binding);

  auto updater = UpdateDFB(prev_dfb_ptr, dfb_);
  root_fn_ = Downcast<Function>(updater.VisitExpr_(root_fn_.get()));

  for (const VarNode* v : used_vars) to_users_.Get(GetRef<Var>(v)).value().push_back(var);

  used_names_.insert(var->name_hint());  // add to used_names_
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_add_binding")
    .set_body_typed([](DataflowBlockRewrite rwt, Binding vb) { rwt->Add(vb); });

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_add")
    .set_body_typed([](DataflowBlockRewrite rwt, Expr expr, Optional<String> name, bool is_dfvar) {
      if (name.get()) {
        rwt->Add(name.value(), expr, is_dfvar);
      } else {
        rwt->Add(expr, is_dfvar);
      }
    });

void DataflowBlockRewriteNode::RemoveUnused(Var unused) {
  // first need to check if this var is used.
  ICHECK(to_users_.count(unused)) << "Cannot remove " << unused << " as it's not found in " << dfb_;
  ICHECK(to_users_[unused].empty()) << unused << " is used by " << to_users_[unused];

  auto prev_dfb_ptr = dfb_.get();

  for (auto it = dfb_->bindings.begin(); it != dfb_->bindings.end(); ++it) {
    // Hope we can simplify this with C++17 one day.
    const VarBindingNode* varbind = (*it).as<VarBindingNode>();
    const MatchShapeNode* mshape = (*it).as<MatchShapeNode>();

    if ((varbind && unused == varbind->var) || (mshape && unused == mshape->var)) {
      dfb_.object.CopyOnWrite()->bindings.erase(it);
      break;
    }
  }

  auto updater = UpdateDFB(prev_dfb_ptr, dfb_);
  root_fn_ = Downcast<Function>(updater.VisitExpr_(root_fn_.get()));

  to_users_.erase(unused);                 // update use-def chain.
  used_names_.erase(unused->name_hint());  // remove from used_names_
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_remove_unused")
    .set_body_typed([](DataflowBlockRewrite rwt, Var unused) { rwt->RemoveUnused(unused); });

IRModule DataflowBlockRewriteNode::MutateIRModule(IRModule irmod) {
  BlockBuilder builder = BlockBuilder::Create(irmod);

  for (auto& p : irmod->functions) {
    if (original_fn_ptr_ == p.second.get()) {
      builder->UpdateFunction(p.first, root_fn_);
      break;
    }
  }

  return builder->GetContextIRModule();
}

TVM_REGISTER_GLOBAL("relax.dfb_rewrite_mutate_irmodule")
    .set_body_typed([](DataflowBlockRewrite rwt, IRModule irmod) {
      return rwt->MutateIRModule(irmod);
    });

}  // namespace relax
}  // namespace tvm
