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
 * \file src/relax/transform/merge_composite_functions.cc
 * \brief Group one or multiple composite functions created by FuseOpsByPattern into a new
 * function.
 *
 * The new function will be annotated with kCodegen and kGlobalSymbol attributes, and it is
 * intented to be offloaded to an external backend.
 *
 * A group for one composite function can be merged into another group for one of its arguments,
 * which we call the parent group for brevity, if the following conditions are met:
 * - The argument is the result of calling a composite function offloaded to the same backend
 * - Merging into the parent group would not create a cyclic dependency with other parent groups
 *
 * For example, in the subgraph below the bottom group cannot be merged into the left parent group,
 * since the right parent group for X depends on an output from the left parent group.
 *
 *  O = Offloaded to A
 *  X = Offloaded to B
 *
 * Correct partitioning:
 *
 *     O         O
 *    / \       /	      \
 *   O   X --> O    +     +    X
 *    \ /             \ /
 *     O               O
 *
 * The algorithm proceeds by assigning a "label", consisting of a pointer to the representative
 * group and the name of the target backend, to each subexpression in the function according to
 * its dataflow. On encountering a call node whose callee is a composite function, we check the
 * two conditions above to see if we can merge this call node into one of its parent groups.
 *
 * To detect cyclic dependencies between groups, we propagate dependency relations, both direct
 * and indirect ones, as we flow through the function. The propagation of indirect dependencies
 * is important since the dependency relation is transitive.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/function.h>

#include <optional>

#include "../../support/arena.h"
#include "utils.h"

namespace tvm {
namespace relax {

using relay::GraphPartitioner;

namespace {

/*! \brief Extract the target name from the given composite name, for example, "dnnl.conv2d_relu" ->
 * "dnnl" */
static String GetCodegenName(const std::string& composite_name) {
  auto delim_pos = composite_name.find(".");
  ICHECK(delim_pos != std::string::npos) << "The pattern name for a composite function should "
                                            "start with a compiler name followed by period.";
  return composite_name.substr(0, delim_pos);
}

/*! \brief A label for a group of composite functions, consisting of the representative group and
 * the target backend name */
struct CompositeGroup {
  GraphPartitioner::Group* representative;
  std::string target;
};

/*! \brief Assign a "CompositeGroup" label to each subexpression in a function according to its
 * dataflow, and returns a mapping from a subexpression to its representative group. */
class CompositeGroupsBuilder : public MemoizedExprTranslator<CompositeGroup> {
 public:
  using Group = GraphPartitioner::Group;
  using GroupMap = std::unordered_map<const Object*, Group*>;
  using MemoizedExprTranslator<CompositeGroup>::VisitExpr_;

  CompositeGroupsBuilder(IRModule mod, support::Arena* arena)
      : mod_(mod), arena_(arena), default_group_(CompositeGroup{nullptr, kDefaultTarget}) {}

  GroupMap Run(Function func) {
    for (const auto& param : func->params) {
      memo_[param] = CompositeGroup{nullptr, kDefaultTarget};
    }
    VisitExpr(func->body);

    GroupMap group_map;
    for (const auto& [expr, group] : memo_) {
      if (group.representative) {
        group_map[expr.get()] = group.representative;
      } else {
        group_map[expr.get()] = arena_->make<Group>();
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
    return CompositeGroup{nullptr, kDefaultTarget};
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
    // Only a call to a composite function is relevant.
    if (auto composite_name = GetCompositeName(call->op)) {
      // Designate one of the parent groups as the "representative" group.
      auto rep_group = GetRepresentative(call->args, *composite_name);

      if (rep_group->num_nodes != 0) {
        // Merge other parent groups into the representative group.
        for (const auto& arg : call->args) {
          auto& arg_group = memo_[arg];
          if (arg_group.target == composite_name && arg_group.representative != rep_group) {
            rep_group->num_nodes += arg_group.representative->num_nodes;
            arg_group.representative->num_nodes = 0;
            arg_group.representative = rep_group;
          }
        }
      }

      // Merge this call node into the representative group.
      ++rep_group->num_nodes;
      return CompositeGroup{rep_group, *composite_name};
    }
    return default_group_;
  }

 private:
  std::optional<std::string> GetCompositeName(const Expr& callee) {
    auto const* gvar = callee.as<GlobalVarNode>();
    if (!gvar) {
      return std::nullopt;
    }

    auto composite_name_opt =
        mod_->Lookup(GetRef<GlobalVar>(gvar))->GetAttr<String>(attr::kComposite);
    if (!composite_name_opt) {
      return std::nullopt;
    }

    return GetCodegenName(composite_name_opt.value());
  }

  Group* GetRepresentative(const Array<Expr>& args, const std::string& composite_name) {
    Group* rep = nullptr;
    std::unordered_set<Group*> parent_deps;

    // Collect groups that parent groups depend on
    for (const auto& arg : args) {
      for (auto parent_dep : group_deps_[memo_[arg].representative]) {
        parent_deps.insert(parent_dep);
      }
    }

    for (const auto& arg : args) {
      auto arg_group = memo_[arg];
      if (arg_group.target == composite_name && !parent_deps.count(arg_group.representative)) {
        // If there is a parent group with the same target, which none of the parent dependency
        // groups depends on, merging "this" call node into the parent group will not form a cyclic
        // dependency.
        rep = arg_group.representative;
      }
    }

    if (rep == nullptr) {
      // If we do not find a valid representative parent group, make a new group.
      // This can happen if all arguments are function parameters or belong to other targets.
      rep = arena_->make<Group>();
      // Set num_nodes to 0 to signify that this representative groups has been newly created.
      rep->num_nodes = 0;
    }

    // Record direct parent dependencies.
    for (const auto& arg : args) {
      auto arg_group = memo_[arg];
      if (arg_group.target != composite_name) {
        group_deps_[rep].insert(arg_group.representative);
      }
    }

    // Propagate parent dependencies.
    for (auto parent_dep : parent_deps) {
      group_deps_[rep].insert(parent_dep);
    }

    return rep;
  }

  const std::string kDefaultTarget = "default";
  IRModule mod_;
  support::Arena* arena_;
  CompositeGroup default_group_;
  std::unordered_map<Group*, std::unordered_set<Group*>> group_deps_;
};

/*! \brief Inline definitions of composite functions at the global level into their call sites.
  This is necessary to make functions created by MergeCompositeFunctions self-contained - each
  external backend compiler does not need to refer to the original containing module.
 */
class CompositeInliner : public ExprMutator {
 public:
  explicit CompositeInliner(IRModule mod) : ExprMutator(mod), mod_(mod) {}
  using ExprMutator::VisitExpr_;

  std::pair<Function, std::string> Run(Function func) {
    target_name_ = "";
    auto new_body = VisitExpr(func->body);
    ICHECK(!target_name_.empty());
    auto new_func =
        Function(func->params, new_body, func->ret_struct_info, func->attrs, func->span);
    return {new_func, target_name_};
  }

  Expr VisitExpr_(const CallNode* call) {
    if (call->op->IsInstance<GlobalVarNode>()) {
      auto gvar = Downcast<GlobalVar>(call->op);
      auto func = CopyWithNewParams(Downcast<Function>(mod_->Lookup(gvar)));
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

IRModule MergeCompositeFunctions(IRModule mod) {
  auto gvar = mod->GetGlobalVar("main");
  auto func = Downcast<Function>(mod->Lookup(gvar));
  support::Arena arena;
  auto group_map = CompositeGroupsBuilder(mod, &arena).Run(func);
  auto new_mod = MakeGroupedFunctions(mod, group_map, true);

  CompositeInliner inliner(mod);
  for (const auto& [gvar, func] : new_mod->functions) {
    if (!mod->functions.count(gvar)) {
      auto [new_func, target_name] = inliner.Run(Downcast<Function>(func));
      new_func = WithAttrs(new_func, {{tvm::attr::kGlobalSymbol, gvar->name_hint},
                                      {attr::kCodegen, String(target_name)}});
      new_mod->Update(gvar, new_func);
    }
  }
  // TODO(@tvm-team): Implicit pass dependency. Revisit when we have a better way to handle this.
  return RemoveUnusedFunctions(new_mod, {"main"});
}

namespace transform {

Pass MergeCompositeFunctions() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule mod, PassContext pc) { return relax::MergeCompositeFunctions(mod); };
  return CreateModulePass(/*pass_function=*/pass_func,       //
                          /*opt_level=*/0,                   //
                          /*pass_name=*/"FuseOpsByPattern",  //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.MergeCompositeFunctions")
    .set_body_typed(MergeCompositeFunctions);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
