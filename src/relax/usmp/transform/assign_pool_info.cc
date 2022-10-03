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

#include <tvm/target/target.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/usmp/utils.h>

#include <string>
#include <utility>

#include "tvm/relax/attrs/memory.h"
#include "tvm/relax/expr_functor.h"

namespace tvm {

/*! \brief Assign PoolInfo objects to allocate that does not have any.
 * The schedulers have the oppurtunity to assign PoolInfo objects to
 * allocate nodes. However, each allocate node is expected to have
 * at least one PoolInfo node assigned to it. If it was not the case,
 * this Pass will assign all PoolInfo objects that the target could
 * access.*/

namespace tir {
namespace usmp {

class TIRPoolInfoAssigner : public StmtExprMutator {
 public:
  explicit TIRPoolInfoAssigner(PrimFunc func, const Map<String, Array<PoolInfo>>& target_pool_infos,
                               const Map<String, Array<PoolInfo>>& target_const_pool_infos)
      : func_(std::move(func)),
        target_pool_infos_(target_pool_infos),
        target_const_pool_infos_(target_const_pool_infos) {}

  Stmt operator()();

 private:
  Stmt VisitStmt_(const AllocateNode* op) override;
  Stmt VisitStmt_(const AllocateConstNode* op) override;

  PrimFunc func_;
  Map<String, Array<PoolInfo>> target_pool_infos_;
  Map<String, Array<PoolInfo>> target_const_pool_infos_;
};

Stmt TIRPoolInfoAssigner::operator()() { return this->VisitStmt(func_->body); }

Stmt TIRPoolInfoAssigner::VisitStmt_(const AllocateNode* op) {
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following PrimFunc does not have a target attr: \n" << func_;
  Map<String, ObjectRef> annotations = Map<String, ObjectRef>(op->annotations);
  if (op->annotations.find(kPoolCandidatesAllocateAttr) == op->annotations.end()) {
    ICHECK(target_pool_infos_.count(tgt.value()->str()) > 0)
        << "Target " << PrettyPrint(tgt) << " not found among " << PrettyPrint(target_pool_infos_);
    annotations.Set(kPoolCandidatesAllocateAttr, target_pool_infos_[tgt.value()->str()]);
  }
  Stmt body = VisitStmt(op->body);
  auto allocate =
      Allocate(op->buffer_var, op->dtype, op->extents, op->condition, body, annotations);
  return std::move(allocate);
}

Stmt TIRPoolInfoAssigner::VisitStmt_(const AllocateConstNode* op) {
  if (!target_const_pool_infos_.size()) {
    return StmtExprMutator::VisitStmt_(op);
  }
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following PrimFunc does not have a target attr: \n" << func_;
  Map<String, ObjectRef> annotations = Map<String, ObjectRef>(op->annotations);
  if (op->annotations.find(kPoolCandidatesAllocateAttr) == op->annotations.end()) {
    annotations.Set(kPoolCandidatesAllocateAttr, target_const_pool_infos_[tgt.value()->str()]);
    annotations.Set(kTargetPoolReadOnlyAccess, Integer(1));
  }
  Stmt body = VisitStmt(op->body);
  auto allocate_const =
      AllocateConst(op->buffer_var, op->dtype, op->extents, op->data, body, annotations);
  return std::move(allocate_const);
}

}  // namespace usmp
}  // namespace tir

namespace relax {
namespace usmp {

class RelaxPoolInfoAssigner : public ExprMutator {
 public:
  explicit RelaxPoolInfoAssigner(Function func,
                                 const Map<String, Array<PoolInfo>>& target_pool_infos,
                                 const Map<String, Array<PoolInfo>>& target_const_pool_infos)
      : func_(std::move(func)),
        target_pool_infos_(target_pool_infos),
        target_const_pool_infos_(target_const_pool_infos) {}

  Expr operator()();

 private:
  Expr VisitExpr_(const CallNode* op) override;

  Function func_;
  Map<String, Array<PoolInfo>> target_pool_infos_;
  Map<String, Array<PoolInfo>> target_const_pool_infos_;
};

Expr RelaxPoolInfoAssigner::operator()() { return this->VisitExpr(func_->body); }

Expr RelaxPoolInfoAssigner::VisitExpr_(const CallNode* call) {
  Expr expr = VisitExprPostOrder_(call);
  call = expr.as<CallNode>();

  static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
  if (call->op != alloc_tensor_op) {
    return GetRef<Call>(call);
  }
  Optional<Target> tgt = func_->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(tgt) << "The following Func does not have a target attr: \n" << func_;
  auto alloc_attrs = call->attrs.as<AllocTensorAttrs>();
  ICHECK(alloc_attrs != nullptr) << "must be AllocTensorAttrs";
  if (alloc_attrs->candidate_memory_pools.size() > 0) {
    return GetRef<Call>(call);
  }
  ICHECK(target_pool_infos_.count(tgt.value()->str()) > 0)
      << "Target " << PrettyPrint(tgt) << " not found among " << PrettyPrint(target_pool_infos_);
  auto alloc_tensor_attr = make_object<AllocTensorAttrs>();
  alloc_tensor_attr->dtype = alloc_attrs->dtype;
  alloc_tensor_attr->runtime_device_index = alloc_attrs->runtime_device_index;
  alloc_tensor_attr->candidate_memory_pools = target_pool_infos_[tgt.value()->str()];
  auto allocate_call =
      Call(call->op, call->args, Attrs(alloc_tensor_attr), call->type_args, call->span);
  return std::move(allocate_call);
}

}  // namespace usmp
}  // namespace relax

class PoolInfoAssigner {
 public:
  explicit PoolInfoAssigner(const IRModule& module) {
    auto main_func =
        Downcast<relax::Function>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
    ICHECK(main_func.defined()) << "main function is not in the module";
    Optional<Target> target_host = main_func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target_host) << "main function does not have a target attr";
    WorkspaceMemoryPools workspace_pools =
        module->GetAttr<WorkspaceMemoryPools>(tvm::attr::kWorkspaceMemoryPools)
            .value_or(WorkspaceMemoryPools({CreateDefaultWorkspaceMemoryPool(module)}));
    // make default ConstantPoolInfo if no constant and no workspace pool infos supplied
    ConstantMemoryPools constant_pools =
        module->GetAttr<ConstantMemoryPools>(tvm::attr::kConstantMemoryPools)
            .value_or(
                module->GetAttr<WorkspaceMemoryPools>(tvm::attr::kWorkspaceMemoryPools).defined()
                    ? ConstantMemoryPools()
                    : ConstantMemoryPools({CreateDefaultConstantMemoryPool(module)}));
    auto to_map = [](auto pool_infos) {
      Map<String, Array<PoolInfo>> pool_map;
      for (const PoolInfo& pool_info : pool_infos) {
        for (const auto& tgt : pool_info->targets) {
          if (pool_map.find(tgt->str()) == pool_map.end()) {
            pool_map.Set(tgt->str(), Array<PoolInfo>());
          }
          Array<PoolInfo> pool_info_arr = pool_map[tgt->str()];
          pool_info_arr.push_back(pool_info);
          pool_map.Set(tgt->str(), pool_info_arr);
        }
      }
      return pool_map;
    };

    target_pool_infos_ = to_map(workspace_pools->pools);
    if (constant_pools.defined()) {
      target_const_pool_infos_ = to_map(constant_pools->pools);
    }
    mod_ = module->ShallowCopy();
  }

  IRModule operator()();

 private:
  IRModule mod_;
  Map<String, Array<PoolInfo>> target_pool_infos_;
  Map<String, Array<PoolInfo>> target_const_pool_infos_;
  WorkspacePoolInfo CreateDefaultWorkspaceMemoryPool(const IRModule& module);
  ConstantPoolInfo CreateDefaultConstantMemoryPool(const IRModule& module) {
    auto p = CreateDefaultWorkspaceMemoryPool(module);
    return ConstantPoolInfo(
        "global_const_workspace", {p->targets}, {},
        PoolInfoProperties(kUnrestrictedPoolSizeHint, kUnknownClockFrequency, kUnknownReadBandwidth,
                           kUnknownWriteBandwidth, 0, 0, {p->target_burst_bytes}, Bool(true)));
  }
};

WorkspacePoolInfo PoolInfoAssigner::CreateDefaultWorkspaceMemoryPool(const tvm::IRModule& module) {
  VLOG(1) << "Creating default memory pool for:" << std::endl << PrettyPrint(module);
  Map<Target, String> target_access;
  auto main_func = Downcast<tvm::BaseFunc>(module->Lookup(::tvm::runtime::symbol::tvm_module_main));
  Target target_host = main_func->GetAttr<Target>(tvm::attr::kTarget).value();
  for (const auto& kv : module->functions) {
    BaseFunc func = kv.second;
    Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
    target_access.Set(target.value_or(target_host), kTargetPoolReadWriteAccess);
  }
  Array<Target> targets;
  for (const auto& kv : target_access) {
    bool exist = false;
    // Exclude targets with the same string representation
    for (const auto& t : targets) {
      if (t->str() == kv.first->str()) {
        exist = true;
      }
    }
    if (!exist) {
      targets.push_back(kv.first);
    }
  }
  return WorkspacePoolInfo(
      "global_workspace", targets,
      PoolInfoProperties(kUnrestrictedPoolSizeHint, kUnknownClockFrequency, kUnknownReadBandwidth,
                         kUnknownWriteBandwidth, 0, 0, {{target_host, 1}}, Bool(true)));
}

IRModule PoolInfoAssigner::operator()() {
  for (const auto& kv : mod_->functions) {
    GlobalVar gv = kv.first;
    if (kv.second->IsInstance<relax::FunctionNode>()) {
      using RelaxPoolInfoAssigner = relax::usmp::RelaxPoolInfoAssigner;
      using Function = relax::Function;
      auto func = runtime::Downcast<Function>(kv.second);
      RelaxPoolInfoAssigner relax_pool_info_assigner =
          RelaxPoolInfoAssigner(func, target_pool_infos_, target_const_pool_infos_);
      relax::Expr body = relax_pool_info_assigner();
      Function new_relax_func =
          Function(func->params, body, func->ret_type, func->ret_shape, func->attrs, func->span);
      mod_->Update(gv, new_relax_func);
    } else if (kv.second->IsInstance<tir::PrimFuncNode>()) {
      using TIRPoolInfoAssigner = tir::usmp::TIRPoolInfoAssigner;
      using PrimFunc = tir::PrimFunc;
      auto func = Downcast<PrimFunc>(kv.second);
      TIRPoolInfoAssigner tir_pool_info_assigner =
          TIRPoolInfoAssigner(func, target_pool_infos_, target_const_pool_infos_);
      tir::Stmt body = tir_pool_info_assigner();
      PrimFunc new_prim_func = PrimFunc(func->params, body, func->ret_type, func->buffer_map,
                                        func->preflattened_buffer_map, func->attrs);
      mod_->Update(gv, new_prim_func);
    }
  }
  return mod_;
}

namespace transform {

tvm::transform::Pass AssignPoolInfo() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return PoolInfoAssigner(m)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "relax.usmp.AssignPoolInfo", {});
}

TVM_REGISTER_GLOBAL("relax.transform.AssignPoolInfo").set_body_typed(AssignPoolInfo);

}  // namespace transform
}  // namespace tvm
