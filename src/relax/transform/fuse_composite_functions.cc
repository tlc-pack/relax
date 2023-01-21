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

/*! TODO
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

#include "../../support/arena.h"
#include "utils.h"

namespace tvm {
namespace relax {

using relay::GraphPartitioner;

namespace {

String GetCodegenName(const std::string& composite_name) {
  auto delim_pos = composite_name.find(".");
  ICHECK(delim_pos != std::string::npos) << "The pattern name for a composite function should "
                                            "start with a compiler name followed by period.";
  return composite_name.substr(0, delim_pos);
}

struct CompositeGroup {
  GraphPartitioner::Group* representative;
  std::string target;
};

class BuildCompositeGroups : public MemoizedExprTranslator<CompositeGroup> {
 public:
  using Group = GraphPartitioner::Group;
  using GroupMap = std::unordered_map<const Object*, Group*>;
  using MemoizedExprTranslator<CompositeGroup>::VisitExpr_;

  BuildCompositeGroups(IRModule mod, support::Arena* arena)
      : mod_(mod), arena_(arena), default_group_(CompositeGroup{nullptr, kDefaultTarget}) {}

  GroupMap Run(Function func) {
    for (const auto& param : func->params) {
      memo_[param] = CompositeGroup{nullptr, kAny};
    }
    VisitExpr(func->body);

    GroupMap group_map;
    for (const auto& [expr, group] : memo_) {
      if (group.representative) {
        group_map[expr.get()] = group.representative;
      } else {
        group_map[expr.get()] = arena_->make<GraphPartitioner::Group>();
      }
    }

    return group_map;
  }

  CompositeGroup VisitBinding(const Binding& binding) {
    if (const auto* node = binding.as<VarBindingNode>()) {
      return VisitBinding_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
    }
  }

  CompositeGroup VisitBindingBlock_(const BindingBlockNode* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
    return default_group_;
  }

  CompositeGroup VisitBindingBlock_(const DataflowBlockNode* block) {
    for (Binding binding : block->bindings) {
      VisitBinding(binding);
    }
    return CompositeGroup{nullptr, kAny};
  }

  CompositeGroup VisitBindingBlock(const BindingBlock& block) {
    if (const auto* node = block.as<DataflowBlockNode>()) {
      return VisitBindingBlock_(node);
    } else if (const auto* node = block.as<BindingBlockNode>()) {
      return VisitBindingBlock_(node);
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << block->GetTypeKey();
    }
  }

  CompositeGroup VisitExpr_(const SeqExprNode* op) {
    for (BindingBlock block : op->blocks) {
      VisitBindingBlock(block);
    }
    return VisitExpr(op->body);
  }

  CompositeGroup VisitExpr_(const CallNode* call) {
    auto const* gvar = call->op.as<GlobalVarNode>();
    if (!gvar) {
      return default_group_;
    }

    auto composite_name_opt =
        mod_->Lookup(GetRef<GlobalVar>(gvar))->GetAttr<String>(attr::kComposite);
    if (!composite_name_opt) {
      return default_group_;
    }

    std::string composite_name = GetCodegenName(composite_name_opt.value());

    auto rep_group = GetRepresentative(call->args, composite_name);

    if (rep_group->num_nodes != 0) {
      for (const auto& arg : call->args) {
        auto& arg_group = memo_[arg];
        if (arg_group.target == composite_name && arg_group.representative != rep_group) {
          rep_group->num_nodes += arg_group.representative->num_nodes;
          arg_group.representative->num_nodes = 0;
          arg_group.representative = rep_group;
        }
      }
    }

    ++rep_group->num_nodes;
    return CompositeGroup{rep_group, composite_name};
  }

 private:
  Group* GetRepresentative(const Array<Expr>& args, const std::string& composite_name) {
    Group* rep = nullptr;
    std::unordered_set<Group*> parent_deps;

    for (const auto& arg : args) {
      for (auto parent_dep : group_deps_[memo_[arg].representative]) {
        parent_deps.insert(parent_dep);
      }
    }

    for (const auto& arg : args) {
      auto arg_group = memo_[arg];
      if (arg_group.target == composite_name && !parent_deps.count(arg_group.representative)) {
        rep = arg_group.representative;
      }
    }

    if (rep == nullptr) {
      rep = arena_->make<Group>();
      rep->num_nodes = 0;
    }

    for (const auto& arg : args) {
      auto arg_group = memo_[arg];
      if (arg_group.target != composite_name) {
        group_deps_[rep].insert(arg_group.representative);
      }
    }

    for (auto parent_dep : parent_deps) {
      group_deps_[rep].insert(parent_dep);
    }

    return rep;
  }

  const std::string kDefaultTarget = "default";
  const std::string kAny = "any";
  IRModule mod_;
  support::Arena* arena_;
  CompositeGroup default_group_;
  std::unordered_map<Group*, std::unordered_set<Group*>> group_deps_;
};

class InlineComposite : public ExprMutator {
 public:
  explicit InlineComposite(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitExpr_;

  Function Run(Function func) {
    target_name_ = "";
    auto new_body = VisitExpr(func->body);
    ICHECK(!target_name_.empty());
    return WithAttr(
        Function(func->params, new_body, func->ret_struct_info, func->attrs, func->span),
        attr::kCodegen, target_name_);
  }

  Expr VisitExpr_(const CallNode* call) {
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto gvar = Downcast<GlobalVar>(call->op);
      auto func = Downcast<Function>(mod_->Lookup(gvar));
      auto composite_name_opt = func->GetAttr<String>(attr::kComposite);
      ICHECK(composite_name_opt);
      std::string composite_name = composite_name_opt.value();
      auto tgt_name = GetCodegenName(composite_name);
      if (!target_name_.empty()) {
        ICHECK(tgt_name == target_name_);
      } else {
        target_name_ = tgt_name;
      }
      return Call(func, call->args);
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  IRModule mod_;
  String target_name_;
};

}  // namespace

IRModule FuseCompositeFunctions(IRModule mod) {
  auto gvar = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(gvar));
  support::Arena arena;
  auto group_map = BuildCompositeGroups(mod, &arena).Run(func);
  auto new_mod = MakeGroupedFunctions(mod, group_map, true);

  InlineComposite postproc(mod);
  for (const auto& [gvar, func] : new_mod->functions) {
    if (!mod->functions.count(gvar)) {
      auto new_func = postproc.Run(Downcast<Function>(func));
      new_mod->Update(gvar, WithAttr(new_func, tvm::attr::kGlobalSymbol, gvar->name_hint));
    }
  }

  return RemoveUnusedFunctions(new_mod, {"main"});
}

namespace transform {

Pass FuseCompositeFunctions() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule mod, PassContext pc) { return relax::FuseCompositeFunctions(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,       //
                          /*opt_level=*/0,                   //
                          /*pass_name=*/"FuseOpsByPattern",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.FuseCompositeFunctions")
    .set_body_typed(FuseCompositeFunctions);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
