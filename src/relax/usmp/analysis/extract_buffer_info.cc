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
 * \file relax/usmp/analysis/extract_buffer_info.cc
 *
 * \brief This analysis pass consumes a TIR IRModule with a main function
 * that defines a ordering in the callees to operators and produces BufferInfo
 * objects that contains information about tir.allocate nodes and liveness
 * conflicts between other tir.allocate nodes.
 */
#include <tvm/relax/expr.h>
#include <tvm/relay/executor.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/usmp/utils.h>
#include <unistd.h>

#include <stack>

#include "../../../runtime/thread_storage_scope.h"
#include "tvm/relax/attrs/memory.h"
#include "tvm/relax/expr_functor.h"
#include "tvm/tir/builtin.h"
#include "tvm/tir/function.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"

namespace tvm {

namespace tir::usmp {
class TIRInfoExtractor;
}

namespace relax::usmp {
class RelaxInfoExtractor;
}

class BufferInfoExtractor;

namespace usmp {
/*!
 * \brief The class to keep buffer information used by this pass.
 *
 * The Relax and TIR visitors would initiate the traversal from the main Relax
 * function and visit into the operator PrimFuncs. They will
 * create unique BufferInfo objects for each Relax/TIR allocation.
 *
 * Every time the buffer variable of an allocation is referenced
 * it will be recorded using the stmt index. However, note that
 * the same buffer variable could be references multiple times
 * from different calls. Thereafter, a sweep is done on all the
 * BufferInfo objects using the per-call liveness events. In the sweep,
 * The BufferInfo objects that are live together will be recorded as
 * mutual conflicts of each other.
 */
class BufferInfoPassData {
  using BufferInfo = tir::usmp::BufferInfo;
  using Call = tir::Call;
  using PrimFunc = tir::PrimFunc;
  using For = tir::For;
  using Allocate = tir::Allocate;
  using AllocateConst = tir::AllocateConst;
  using Function = relax::Function;

  /*!
   * \brief Maintains the mapping of BufferInfo to their associated TIR Statements or Relax Expr.
   */
  Map<BufferInfo, runtime::ObjectRef> buffer_info_map_;
  /*!
   * \brief Records the order of calls in the main for stability.
   */
  std::vector<BaseExpr> call_order_;
  /*!
   * \brief Lookup to avoid adding duplicates to `call_order_`.
   */
  std::unordered_set<BaseExpr, ObjectPtrHash, ObjectPtrEqual> call_order_contents_;
  /*!
   * \brief Records first access in-terms of TIR Stmts/Relax Expr to each buffer per call
   *
   * This is because multiple calls could happen to the same PrimFunc.
   */
  std::unordered_map<BaseExpr, Map<runtime::ObjectRef, Integer>, ObjectPtrHash, ObjectPtrEqual>
      buffer_info_start_stmt_idx_;
  /*!
   * \brief Records last access in-terms of TIR Stmts/Relax Expr to each buffer per call
   *
   * This is because multiple calls could happen to the same PrimFunc.
   */
  std::unordered_map<BaseExpr, Map<runtime::ObjectRef, Integer>, ObjectPtrHash, ObjectPtrEqual>
      buffer_info_end_stmt_idx_;
  /*!
   * \brief This structure contains information regarding a TIR Allocate node / Relax call node to
   * alloc_tensor.
   */
  struct AllocateInfo {
    runtime::ObjectRef Allocate;
    BaseFunc func;
    BaseExpr call;
  };

  /*!
   * \brief Maintains the mapping of TIR buffer variable / Relax Var to their allocate infos to
   * ensure that only one BufferInfo object is created.
   */
  std::unordered_map<BaseExpr, AllocateInfo, ObjectPtrHash, ObjectPtrEqual> allocate_infos;
  /*!
   * \brief Indicates a count of stmts visited so far to use as a metric of liveness
   */
  int current_stmt_idx_ = 0;
  /*!
   * \brief This structure is supposed to contain information around the scope
   * the visitor is currently in.
   */
  struct ScopeInfo {
    /*!
     * \brief We need to record access per call
     */
    BaseExpr call;
    /*!
     * \brief Having access to PrimFunc/RelaxFunc metadata is useful
     */
    BaseFunc func;
    /*!
     * \brief We currently support only serial for loops. Therefore
     * need to know what kind of for loop the visitor is in. Only used when visiting PrimFuncs.
     */
    For for_loop;
    /*!
     * \brief We record the live TIR allocate_nodes and Relax allocate Expr because once in loops
     * the liveness range has to be extended to the whole of the nested
     * loops structure.
     */
    std::unordered_set<runtime::ObjectRef, ObjectPtrHash, ObjectPtrEqual> allocate_nodes;
    /*
     * \brief We record the live allocate_const_nodes because once in loops
     * the liveness range has to be extended to the whole of the nested
     * loops structure.
     */
    std::unordered_set<AllocateConst, ObjectPtrHash, ObjectPtrEqual> allocate_const_nodes;
    /*!
     * \brief This is recorded to extend the liveness of all allocates within
     * nested loop structure. Only used for PrimFuncs.
     */
    Integer initial_stmt_of_the_nested_loops;
  };
  std::stack<ScopeInfo> scope_stack_;

  /*!
   * \brief A liveness event tracks when
   * traversing the tir.Stmts/Relax.Expr where allocations
   * begin or cease to be Live. This particular struct
   * is used to solve interval overlap problem using
   * a sweep-line algorithm. For that, we need to record
   * where the liveness event occurred in a chronological
   * order.
   */
  enum LivenessEventType { START = 0, END = 1 };
  struct LivenessEvent {
    size_t tick;
    LivenessEventType le_type;
    BufferInfo buffer_info;
    bool operator==(const LivenessEvent& other) {
      if (tick == other.tick && le_type == other.le_type && buffer_info == other.buffer_info) {
        return true;
      }
      return false;
    }
  };
  /*!
   * \brief We need to create unique buffer name is the same name is used in
   * two allocate nodes for clarity for memory planning algorithms.
   */
  std::string GetUniqueBufferName(std::string name);

  /*!
   * \brief This is per buffer name counter to aid the generating the above
   * unique name.
   */
  std::unordered_map<std::string, int> buffer_names;
  /*!
   * \brief The Relax main function calls to external functions to be able to
   * support BYOC. Therefore, this Map records functions that are present
   * in the IRModule by name/
   */
  Map<String, BaseFunc> functions_;

  /*!
   * \brief The IRModule being analyzed.
   */
  IRModule module_;

  friend class tvm::BufferInfoExtractor;
  friend class tir::usmp::TIRInfoExtractor;
  friend class relax::usmp::RelaxInfoExtractor;
};

}  // namespace usmp

namespace tir {
namespace usmp {

class TIRInfoExtractor : public StmtExprVisitor {
  using BufferInfoPassData = tvm::usmp::BufferInfoPassData;

 public:
  explicit TIRInfoExtractor(BufferInfoPassData& pass_data) : pass_data_(pass_data) {}

  void VisitPrimFunc(const PrimFunc& func, const BaseExpr& call);
  void UpdateAliases(const Array<BaseExpr>& args, const PrimFunc& func);

 private:
  void VisitStmt(const Stmt& n) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AllocateConstNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const BufferLoadNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const ForNode* op) override;

  void RecordAllocateNodeInfo(const AllocateNode* op);
  void RecordAllocateConstNodeInfo(const AllocateConstNode* op);

  BufferInfoPassData& pass_data_;
};

void TIRInfoExtractor::VisitStmt(const Stmt& n) {
  pass_data_.current_stmt_idx_ += 1;
  StmtExprVisitor::VisitStmt(n);
}

void TIRInfoExtractor::RecordAllocateNodeInfo(const AllocateNode* op) {
  auto size_bytes = tir::usmp::CalculateExtentsSize(op);
  // We only statically memory plan allocates with known
  // compile time sizes.
  if (size_bytes.defined()) {
    if (pass_data_.allocate_infos.find(op->buffer_var) == pass_data_.allocate_infos.end()) {
      // By default, the core compiler is assumed to attach the a default pool to each allocate.
      ICHECK(op->annotations.count(tir::usmp::kPoolCandidatesAllocateAttr))
          << "Every statically sized allocate node needs an pool candidate attribute";
      auto pool_candidates =
          Downcast<Array<PoolInfo>>(op->annotations[tir::usmp::kPoolCandidatesAllocateAttr]);

      ICHECK(pool_candidates.size() > 0)
          << "The AssignPoolInfo pass should at least attach a single PoolInfo. If there were no "
             "user-given arguments for memory pools, the default behaviour is a single size "
             "un-restricted pool is assigned";
      PrimFunc func = Downcast<PrimFunc>(pass_data_.scope_stack_.top().func);
      Optional<tvm::relay::Executor> executor_config =
          pass_data_.module_->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor);
      Integer workspace_alignment = 16;
      if (executor_config) {
        workspace_alignment =
            executor_config.value()->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
      }

      BufferInfoKind bi_kind = BufferInfoKind::kIntermediate;
      String buffer_info_name = op->buffer_var->name_hint;
      if (op->annotations.find(kInputTensorAllocate) != op->annotations.end()) {
        bi_kind = BufferInfoKind::kInput;
        // using original input name instead of the buffer_var name
        // because this name will be used in the lowering to convey
        // the pool allocation.
        buffer_info_name = Downcast<String>(op->annotations[kInputTensorAllocate]);
      } else if (op->annotations.find(kOutputTensorAllocate) != op->annotations.end()) {
        bi_kind = BufferInfoKind::kOutput;
        // using original output name instead of the buffer_var name
        // because this name will be used in the lowering to convey
        // the pool allocation.
        buffer_info_name = Downcast<String>(op->annotations[kOutputTensorAllocate]);
      }
      auto buffer_info = BufferInfo(pass_data_.GetUniqueBufferName(buffer_info_name), size_bytes,
                                    pool_candidates, workspace_alignment, bi_kind);
      auto allocate = GetRef<Allocate>(op);
      pass_data_.allocate_infos[op->buffer_var] = BufferInfoPassData::AllocateInfo{
          allocate, Downcast<PrimFunc>(pass_data_.scope_stack_.top().func),
          pass_data_.scope_stack_.top().call};
      pass_data_.buffer_info_map_.Set(buffer_info, allocate);
    } else {
      // Update the allocate info with the latest call
      BufferInfoPassData::AllocateInfo ai = pass_data_.allocate_infos[op->buffer_var];
      ai.call = pass_data_.scope_stack_.top().call;
      pass_data_.allocate_infos[op->buffer_var] = ai;
    }
  }
}

void TIRInfoExtractor::VisitStmt_(const AllocateNode* op) {
  using ScopeInfo = BufferInfoPassData::ScopeInfo;
  ScopeInfo& current_scope_info = pass_data_.scope_stack_.top();
  const auto& type = Downcast<PointerType>(op->buffer_var->type_annotation);
  const auto& storage_scope = runtime::StorageScope::Create(type->storage_scope);

  // If the allocate is in a for loop, USMP currently only looks at serial for loops.
  // If its not a serial for loop, then memory planner will omit them in the current memory planning
  // process leaving them to as tir.allocate nodes for codegen. Additionally, the USMP can only work
  // with buffers that have global storage_scope

  if (storage_scope.rank == runtime::StorageRank::kGlobal) {
    if (!current_scope_info.for_loop.defined()) {
      RecordAllocateNodeInfo(op);
    } else if (current_scope_info.for_loop.defined() &&
               current_scope_info.for_loop->kind == ForKind::kSerial) {
      RecordAllocateNodeInfo(op);
    }
  }
  StmtExprVisitor::VisitStmt(op->body);
  current_scope_info.allocate_nodes.erase(GetRef<Allocate>(op));
}

void TIRInfoExtractor::VisitStmt_(const AllocateConstNode* op) {
  using ScopeInfo = BufferInfoPassData::ScopeInfo;
  ScopeInfo& current_scope_info = pass_data_.scope_stack_.top();
  RecordAllocateConstNodeInfo(op);
  StmtExprVisitor::VisitStmt(op->body);
  current_scope_info.allocate_const_nodes.erase(GetRef<AllocateConst>(op));
}

void TIRInfoExtractor::RecordAllocateConstNodeInfo(const AllocateConstNode* op) {
  if (!op->annotations.count(kPoolCandidatesAllocateAttr)) {
    return;
  }
  Integer size_bytes = CalculateExtentsSize(op);
  ICHECK(size_bytes.defined()) << "constant node size should be defined";
  const auto& buffer_var = op->buffer_var;
  if (pass_data_.allocate_infos.find(buffer_var) == pass_data_.allocate_infos.end()) {
    // By default, the core compiler is assumed to attach the a default pool to each allocate.
    ICHECK(op->annotations.count(kPoolCandidatesAllocateAttr))
        << "Every statically sized allocate node needs an pool candidate attribute";
    auto pool_candidates = Downcast<Array<PoolInfo>>(op->annotations[kPoolCandidatesAllocateAttr]);
    ICHECK(pool_candidates.size() > 0)
        << "The core compiler should at least attach a single PoolInfo. If there were no "
           "user-given arguments for memory pools, the default behaviour is a single size "
           "un-restricted pool is assigned";
    PrimFunc func = Downcast<PrimFunc>(pass_data_.scope_stack_.top().func);
    Optional<tvm::relay::Executor> executor_config =
        pass_data_.module_->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor);
    Integer alignment = 16;
    if (executor_config) {
      alignment =
          executor_config.value()->GetAttr<Integer>("constant-byte-alignment").value_or(alignment);
    }
    auto buffer_info = BufferInfo(pass_data_.GetUniqueBufferName(buffer_var->name_hint), size_bytes,
                                  pool_candidates, alignment);
    auto allocate = GetRef<AllocateConst>(op);
    pass_data_.allocate_infos[buffer_var] = BufferInfoPassData::AllocateInfo{
        allocate, Downcast<PrimFunc>(pass_data_.scope_stack_.top().func),
        pass_data_.scope_stack_.top().call};
    pass_data_.buffer_info_map_.Set(buffer_info, allocate);
  } else {
    // Update the allocate info with the latest call
    BufferInfoPassData::AllocateInfo ai = pass_data_.allocate_infos[buffer_var];
    ai.call = pass_data_.scope_stack_.top().call;
    pass_data_.allocate_infos[buffer_var] = ai;
  }
}

void TIRInfoExtractor::VisitStmt_(const ForNode* op) {
  using ScopeInfo = BufferInfoPassData::ScopeInfo;
  ScopeInfo si{pass_data_.scope_stack_.top().call,
               pass_data_.scope_stack_.top().func,
               GetRef<For>(op),
               pass_data_.scope_stack_.top().allocate_nodes,
               pass_data_.scope_stack_.top().allocate_const_nodes,
               pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops};
  if (!pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops.defined()) {
    si.initial_stmt_of_the_nested_loops = Integer(pass_data_.current_stmt_idx_);
  }
  BaseExpr current_call = pass_data_.scope_stack_.top().call;
  auto current_func = pass_data_.scope_stack_.top().func;
  pass_data_.scope_stack_.push(si);
  StmtExprVisitor::VisitStmt_(op);
  // Extending the liveness to beginning of for-loop next and end of the current for-loop
  for (const runtime::ObjectRef& ref : pass_data_.scope_stack_.top().allocate_nodes) {
    BufferInfoPassData::AllocateInfo ai;
    if (ref->IsInstance<relax::VarNode>()) {
      auto expr = runtime::Downcast<BaseExpr>(ref);
      ai = pass_data_.allocate_infos[expr];
    } else if (ref->IsInstance<AllocateNode>()) {
      auto allocate = runtime::Downcast<Allocate>(ref);
      ai = pass_data_.allocate_infos[allocate->buffer_var];
    }
    auto allocate = ref;
    BaseExpr update_call = current_call;
    // If the allocate does not belong to current func
    // We need to update the call to which the allocate belongs to
    if (ai.func != current_func) {
      update_call = ai.call;
    }
    if (pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops->value <
        pass_data_.buffer_info_start_stmt_idx_[update_call][allocate].IntValue()) {
      pass_data_.buffer_info_start_stmt_idx_[update_call].Set(
          allocate, pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops->value);
    }
    if (pass_data_.current_stmt_idx_ >
        pass_data_.buffer_info_end_stmt_idx_[update_call][allocate].IntValue()) {
      pass_data_.buffer_info_end_stmt_idx_[update_call].Set(allocate, pass_data_.current_stmt_idx_);
    }
  }
  pass_data_.scope_stack_.pop();
}

void TIRInfoExtractor::VisitExpr_(const BufferLoadNode* op) {
  this->VisitExpr(op->buffer->data);
  StmtExprVisitor::VisitExpr_(op);
}

void TIRInfoExtractor::VisitStmt_(const BufferStoreNode* op) {
  this->VisitExpr(op->buffer->data);
  StmtExprVisitor::VisitStmt_(op);
}

void TIRInfoExtractor::VisitExpr_(const VarNode* op) {
  auto var = GetRef<Var>(op);
  auto current_call = pass_data_.scope_stack_.top().call;
  auto current_func = pass_data_.scope_stack_.top().func;
  if (pass_data_.allocate_infos.count(var)) {
    auto allocate = pass_data_.allocate_infos[var].Allocate;
    auto allocate_func = pass_data_.allocate_infos[var].func;
    BaseExpr update_call = current_call;
    if (allocate_func != current_func) {
      // If the allocate node does not belong to the current primfunc.
      // It's access should be reported to the call to PrimFunc that
      // Allocate belong to.
      update_call = pass_data_.allocate_infos[var].call;
    }
    if (pass_data_.buffer_info_start_stmt_idx_[update_call].count(allocate) == 0) {
      pass_data_.buffer_info_start_stmt_idx_[update_call].Set(allocate,
                                                              pass_data_.current_stmt_idx_);
    }
    pass_data_.buffer_info_end_stmt_idx_[update_call].Set(allocate, pass_data_.current_stmt_idx_);

    BufferInfoPassData::ScopeInfo& currect_scope_info = pass_data_.scope_stack_.top();
    if (currect_scope_info.for_loop.defined()) {
      if (allocate->IsInstance<AllocateNode>()) {
        currect_scope_info.allocate_nodes.insert(Downcast<Allocate>(allocate));
      } else if (allocate->IsInstance<AllocateConstNode>()) {
        currect_scope_info.allocate_const_nodes.insert(Downcast<AllocateConst>(allocate));
      } else if (allocate->IsInstance<relax::VarNode>()) {
        currect_scope_info.allocate_nodes.insert(Downcast<relax::Var>(allocate));
      } else {
        LOG(FATAL) << "Handling of " << allocate->GetTypeKey() << " is not implemented";
      }
    }
  }
  StmtExprVisitor::VisitExpr_(op);
}

Array<tir::Var> static GetMatchedBuffers(const PrimFunc& func) {
  Array<Var> buffer_vars;
  for (unsigned int i = 0; i < func->params.size() - 1; i++) {
    Var param = func->params[i];
    buffer_vars.push_back(func->buffer_map[param]->data);
  }
  Var last_param = func->params.back();
  // Checks whether last var is present in the buffer map
  // because it could be the resource handle
  if (func->buffer_map.find(last_param) != func->buffer_map.end()) {
    buffer_vars.push_back(func->buffer_map[last_param]->data);
  }
  return buffer_vars;
}

void TIRInfoExtractor::UpdateAliases(const Array<BaseExpr>& args, const PrimFunc& func) {
  auto param_buffers = GetMatchedBuffers(func);
  // Last var could be a resource handle that does not have a Buffer
  ICHECK(args.size() == param_buffers.size() || args.size() - 1 == param_buffers.size());
  for (size_t i = 0; i < param_buffers.size(); i++) {
    auto arg = args[i];
    auto param_buf = param_buffers[i];
    // If tir.allocates are passed in to functions
    // The function params are re-directed to point
    // to the original allocate
    if (arg->IsInstance<LoadNode>()) {
      auto load = Downcast<Load>(arg);
      if (pass_data_.allocate_infos.count(load->buffer_var)) {
        pass_data_.allocate_infos[param_buf] = pass_data_.allocate_infos[load->buffer_var];
      }
    } else if (arg->IsInstance<VarNode>()) {
      auto var = Downcast<Var>(arg);
      if (pass_data_.allocate_infos.count(var)) {
        pass_data_.allocate_infos[param_buf] = pass_data_.allocate_infos[var];
      }
    } else if (arg->IsInstance<relax::VarNode>()) {
      auto var = Downcast<relax::Var>(arg);
      if (pass_data_.allocate_infos.count(var)) {
        pass_data_.allocate_infos[param_buf] = pass_data_.allocate_infos[var];
      }
    }
  }
}

void TIRInfoExtractor::VisitPrimFunc(const PrimFunc& func, const BaseExpr& call) {
  BufferInfoPassData::ScopeInfo si{call,
                                   func,
                                   pass_data_.scope_stack_.top().for_loop,
                                   pass_data_.scope_stack_.top().allocate_nodes,
                                   pass_data_.scope_stack_.top().allocate_const_nodes,
                                   pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops};
  if (pass_data_.call_order_contents_.count(call) == 0) {
    pass_data_.call_order_contents_.insert(call);
    pass_data_.call_order_.push_back(call);
  }
  pass_data_.scope_stack_.push(si);
  this->VisitStmt(func->body);
  pass_data_.scope_stack_.pop();
}

void TIRInfoExtractor::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::call_extern()) || op->op.same_as(builtin::tvm_call_cpacked())) {
    StringImm func_name = Downcast<StringImm>(op->args[0])->value;
    if (pass_data_.functions_.find(func_name->value) != pass_data_.functions_.end()) {
      auto func = pass_data_.functions_.at(func_name->value);
      auto actual_args = Array<BaseExpr>(op->args.begin() + 1, op->args.end());
      this->UpdateAliases(actual_args, Downcast<PrimFunc>(func));
      VisitPrimFunc(Downcast<PrimFunc>(func), GetRef<Call>(op));
      return;
    }
  }
  if (op->op->IsInstance<PrimFuncNode>()) {
    auto func = Downcast<PrimFunc>(op->op);
    auto actual_args = Array<BaseExpr>(op->args.begin(), op->args.end());
    this->UpdateAliases(actual_args, func);
    VisitPrimFunc(func, GetRef<Call>(op));
    return;
  }
  StmtExprVisitor::VisitExpr_(op);
}

}  // namespace usmp
}  // namespace tir

namespace relax {
namespace usmp {

class RelaxInfoExtractor : public relax::ExprVisitor {
  using BufferInfoPassData = tvm::usmp::BufferInfoPassData;

 public:
  explicit RelaxInfoExtractor(BufferInfoPassData& pass_data) : pass_data_(pass_data) {}

  void VisitRelaxFunc(const Function& func, const Call& call);

 private:
  void VisitExpr(const Expr& expr) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const CallNode* op) override;

  void VisitBinding_(const VarBindingNode* binding);

  void VisitAllocTensorVarBinding(const VarBindingNode* op);
  void RecordAllocateNodeInfo(const VarBindingNode* op);

  BufferInfoPassData& pass_data_;
};

void RelaxInfoExtractor::VisitExpr(const Expr& expr) {
  pass_data_.current_stmt_idx_ += 1;
  ExprVisitor::VisitExpr(expr);
}

void RelaxInfoExtractor::VisitExpr_(const VarNode* op) {
  auto var = GetRef<Var>(op);

  BaseExpr current_call = pass_data_.scope_stack_.top().call;
  BaseFunc current_func = pass_data_.scope_stack_.top().func;
  if (pass_data_.allocate_infos.count(var)) {
    auto allocate = pass_data_.allocate_infos[var].Allocate;
    auto allocate_func = pass_data_.allocate_infos[var].func;
    BaseExpr update_call = current_call;
    if (allocate_func != current_func) {
      // If the allocate node does not belong to the current func,
      // the access should be reported to the call to the func that
      // the node belongs to.
      update_call = pass_data_.allocate_infos[var].call;
    }
    if (pass_data_.buffer_info_start_stmt_idx_[update_call].count(allocate) == 0) {
      pass_data_.buffer_info_start_stmt_idx_[update_call].Set(allocate,
                                                              pass_data_.current_stmt_idx_);
    }
    pass_data_.buffer_info_end_stmt_idx_[update_call].Set(allocate, pass_data_.current_stmt_idx_);

    BufferInfoPassData::ScopeInfo& currect_scope_info = pass_data_.scope_stack_.top();
    if (currect_scope_info.for_loop.defined()) {
      currect_scope_info.allocate_nodes.insert(allocate);
    }
  }
  ExprVisitor::VisitExpr_(op);
}

void RelaxInfoExtractor::VisitBinding_(const VarBindingNode* binding) {
  auto node = GetRef<VarBinding>(binding);
  if (node->value->IsInstance<CallNode>()) {
    auto call_node = runtime::Downcast<Call>(node->value);
    static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    if (call_node->op == alloc_tensor_op) {
      VisitAllocTensorVarBinding(binding);
      return;
    }
  } else if (node->value->IsInstance<VarNode>()) {
    // Update the allocate info map with the alias.
    auto aliased_var = runtime::Downcast<Var>(node->value);
    if (pass_data_.allocate_infos.count(aliased_var)) {
      BufferInfoPassData::AllocateInfo ai = pass_data_.allocate_infos[aliased_var];
      ai.call = pass_data_.scope_stack_.top().call;
      pass_data_.allocate_infos[node->var] = ai;
    }
  }
  ExprVisitor::VisitBinding_(binding);
}

static Integer CalculateRelaxExtentsSize(const DataType& dtype, const Array<PrimExpr>& extents) {
  size_t element_size_bytes = dtype.bytes();
  size_t num_elements = 1;
  for (const auto& ext : extents) {
    if (ext->IsInstance<IntImmNode>()) {
      num_elements *= Downcast<IntImm>(ext)->value;
    }
  }
  return Integer(num_elements * element_size_bytes);
}

void RelaxInfoExtractor::RecordAllocateNodeInfo(const VarBindingNode* op) {
  auto var_binding = runtime::GetRef<VarBinding>(op);
  // TODO(gigiblender) checked_type of relax.alloc_tensor should not be dynamic when
  //  constant sizes are used.
  ICHECK(op->var->checked_type()->IsInstance<DynTensorTypeNode>())
      << "Expected a dynamic tensor type object";
  auto dyn_tensor_type = runtime::Downcast<DynTensorType>(op->var->checked_type());
  ICHECK(op->var->shape()->IsInstance<ShapeExprNode>()) << "Expected a ShapeExpr";
  auto shape_expr = runtime::Downcast<ShapeExpr>(op->var->shape());
  auto size_bytes = CalculateRelaxExtentsSize(dyn_tensor_type->dtype, shape_expr->values);
  if (size_bytes.defined()) {
    auto var_node = op->var;
    auto call_node = runtime::Downcast<Call>(op->value);
    if (pass_data_.allocate_infos.find(var_node) == pass_data_.allocate_infos.end()) {
      // By default, the core compiler is assumed to attach the a default pool to each allocate.
      auto call_dict_attrs = call_node->attrs.as<AllocTensorAttrs>();
      auto pool_candidates = call_dict_attrs->candidate_memory_pools;
      ICHECK(pool_candidates.size() > 0)
          << "The AssignPoolInfo pass should at least attach a single PoolInfo. If there were no "
             "user-given arguments for memory pools, the default behaviour is a single size "
             "un-restricted pool is assigned";
      BaseFunc func = pass_data_.scope_stack_.top().func;
      Optional<tvm::relay::Executor> executor_config =
          pass_data_.module_->GetAttr<tvm::relay::Executor>(tvm::attr::kExecutor);
      Integer workspace_alignment = 16;
      if (executor_config) {
        workspace_alignment =
            executor_config.value()->GetAttr<Integer>("workspace-byte-alignment").value_or(16);
      }

      tir::usmp::BufferInfoKind bi_kind = tir::usmp::BufferInfoKind::kIntermediate;
      String buffer_info_name = op->var->name_hint();
      auto buffer_info =
          tir::usmp::BufferInfo(pass_data_.GetUniqueBufferName(buffer_info_name), size_bytes,
                                pool_candidates, workspace_alignment, bi_kind);
      pass_data_.allocate_infos[var_node] = BufferInfoPassData::AllocateInfo{
          var_node, pass_data_.scope_stack_.top().func, pass_data_.scope_stack_.top().call};
      pass_data_.buffer_info_map_.Set(buffer_info, var_node);
    } else {
      // Update the allocate info with the latest call
      BufferInfoPassData::AllocateInfo ai = pass_data_.allocate_infos[var_node];
      ai.call = pass_data_.scope_stack_.top().call;
      pass_data_.allocate_infos[var_node] = ai;
    }
  }
}

void RelaxInfoExtractor::VisitAllocTensorVarBinding(const VarBindingNode* op) {
  BufferInfoPassData::ScopeInfo& current_scope_info = pass_data_.scope_stack_.top();
  RecordAllocateNodeInfo(op);
  ExprVisitor::VisitBinding_(op);
  current_scope_info.allocate_nodes.erase(op->var);
}

void RelaxInfoExtractor::VisitExpr_(const CallNode* op) {
  auto node = GetRef<Call>(op);
  static const Op& alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
  if (op->op == alloc_tensor_op) {
    // Handled by the VarBinding visit method
    return;
  }

  if (op->op->IsInstance<ExternFuncNode>()) {
    String func_name = runtime::Downcast<ExternFunc>(op->op)->global_symbol;
    if (pass_data_.functions_.find(func_name) != pass_data_.functions_.end()) {
      auto func = pass_data_.functions_.at(func_name);
      if (func->IsInstance<tir::PrimFuncNode>()) {
        auto actual_args = Array<BaseExpr>(op->args.begin(), op->args.end());
        tir::usmp::TIRInfoExtractor tir_info_extractor = tir::usmp::TIRInfoExtractor(pass_data_);
        tir_info_extractor.UpdateAliases(actual_args, Downcast<tir::PrimFunc>(func));
        tir_info_extractor.VisitPrimFunc(Downcast<tir::PrimFunc>(func), GetRef<Call>(op));
        return;
      }
    }
  }
  if (op->op->IsInstance<relax::FunctionNode>()) {
    auto func = Downcast<relax::Function>(op->op);
    ICHECK(false) << "Calls to Relax functions are not supported." << PrettyPrint(func);
  }
  if (op->op->IsInstance<GlobalVarNode>()) {
    auto global_var = Downcast<GlobalVar>(op->op);
    ICHECK(false) << "Calls to Relax functions are not supported: " << global_var->name_hint;
  }
  ExprVisitor::VisitExpr_(op);
}

void RelaxInfoExtractor::VisitRelaxFunc(const Function& func, const Call& call) {
  BufferInfoPassData::ScopeInfo si{call,
                                   func,
                                   pass_data_.scope_stack_.top().for_loop,
                                   pass_data_.scope_stack_.top().allocate_nodes,
                                   pass_data_.scope_stack_.top().allocate_const_nodes,
                                   pass_data_.scope_stack_.top().initial_stmt_of_the_nested_loops};
  if (pass_data_.call_order_contents_.count(call) == 0) {
    pass_data_.call_order_contents_.insert(call);
    pass_data_.call_order_.push_back(call);
  }
  pass_data_.scope_stack_.push(si);
  this->VisitExpr(func->body);
  pass_data_.scope_stack_.pop();
}

}  // namespace usmp
}  // namespace relax

class BufferInfoExtractor {
  using BufferInfoPassData = tvm::usmp::BufferInfoPassData;
  using BufferInfoAnalysis = tir::usmp::BufferInfoAnalysis;

 public:
  explicit BufferInfoExtractor(const IRModule& module) {
    pass_data_.module_ = module;
    for (const auto& gv_func : module->functions) {
      if (gv_func.second->IsInstance<tir::PrimFuncNode>() ||
          gv_func.second->IsInstance<relax::FunctionNode>()) {
        pass_data_.functions_.Set(gv_func.first->name_hint, Downcast<BaseFunc>(gv_func.second));
      }
    }
    // Pushing a scope info for the initial body of the main function
    pass_data_.scope_stack_.push(BufferInfoPassData::ScopeInfo());
  }
  BufferInfoAnalysis operator()(const relax::Function& func);

 private:
  BufferInfoPassData pass_data_;
};

std::string usmp::BufferInfoPassData::GetUniqueBufferName(std::string name) {
  if (buffer_names.find(name) == buffer_names.end()) {
    buffer_names[name] = 1;
    return name;
  } else {
    buffer_names[name] = buffer_names[name] + 1;
    return name + std::to_string(buffer_names[name]);
  }
}

tir::usmp::BufferInfoAnalysis BufferInfoExtractor::operator()(const relax::Function& main_func) {
  using LivenessEvent = BufferInfoPassData::LivenessEvent;
  using LivenessEventType = BufferInfoPassData::LivenessEventType;
  using BufferInfo = tir::usmp::BufferInfo;
  using RelaxInfoExtractor = relax::usmp::RelaxInfoExtractor;

  RelaxInfoExtractor relax_info_extractor = RelaxInfoExtractor(pass_data_);
  relax_info_extractor.VisitRelaxFunc(main_func, relax::Call());

  // Create a vector of liveness events
  // associated with each BufferNodes.
  std::vector<LivenessEvent> le_events_timeline;
  for (const auto& kv1 : pass_data_.buffer_info_map_) {
    auto allocate = kv1.second;
    auto buffer_info = Downcast<BufferInfo>(kv1.first);

    ICHECK(pass_data_.call_order_.size() >= pass_data_.buffer_info_end_stmt_idx_.size());

    for (const BaseExpr& call : pass_data_.call_order_) {
      Map<runtime::ObjectRef, Integer> buffer_info_starts =
          pass_data_.buffer_info_start_stmt_idx_[call];
      if (buffer_info_starts.find(allocate) != buffer_info_starts.end()) {
        LivenessEvent le_event_start;
        le_event_start.buffer_info = buffer_info;
        le_event_start.le_type = LivenessEventType::START;
        le_event_start.tick = buffer_info_starts[allocate].IntValue();
        le_events_timeline.push_back(le_event_start);
      }
    }

    for (const BaseExpr& call : pass_data_.call_order_) {
      Map<runtime::ObjectRef, Integer> buffer_info_ends =
          pass_data_.buffer_info_end_stmt_idx_[call];
      if (buffer_info_ends.find(allocate) != buffer_info_ends.end()) {
        LivenessEvent le_event_end;
        le_event_end.buffer_info = buffer_info;
        le_event_end.le_type = LivenessEventType::END;
        le_event_end.tick = buffer_info_ends[allocate].IntValue();
        le_events_timeline.push_back(le_event_end);
      }
    }
  }

  // Sort the liveness events based on the chronological
  // ordering. For events that are simultaneous, START event
  // takes precedence.
  std::sort(le_events_timeline.begin(), le_events_timeline.end(),
            [](const LivenessEvent& lhs, const LivenessEvent& rhs) {
              if (lhs.tick < rhs.tick) {
                return true;
              } else if (lhs.tick == rhs.tick && lhs.le_type == LivenessEventType::START &&
                         rhs.le_type == LivenessEventType::END) {
                return true;
              }
              return false;
            });

  // Traverse the liveness events using a open set to track what
  // is live while updating the conflicts through out the linear traversal
  int open_set_size = 0;
  int max_open_set_size = 0;
  std::unordered_set<BufferInfo, ObjectPtrHash, ObjectPtrEqual> open_set;
  for (const auto& le_event : le_events_timeline) {
    if (le_event.le_type == LivenessEventType::START) {
      for (const BufferInfo& open_buffer_info : open_set) {
        open_buffer_info->conflicts.push_back(le_event.buffer_info);
        if (le_event.buffer_info != open_buffer_info) {
          le_event.buffer_info->conflicts.push_back(open_buffer_info);
        }
      }
      open_set_size += le_event.buffer_info->size_bytes.IntValue();
      if (open_set_size > max_open_set_size) {
        max_open_set_size = open_set_size;
      }
      open_set.insert(le_event.buffer_info);
    } else {
      open_set_size -= le_event.buffer_info->size_bytes.IntValue();
      open_set.erase(le_event.buffer_info);
    }
  }

  // All ConstantPoolInfo items should have conflicts with each other
  // as they will be placed in RO segment and pre-initialized. To achieve this
  // first, split buffers to vars (WorkspacePoolInfo items) and constants (ConstantPoolInfo items):
  Array<BufferInfo> buffer_info_vars;
  Array<BufferInfo> buffer_info_constants;
  for (const auto& kv : this->pass_data_.buffer_info_map_) {
    const auto& stmt = kv.second;
    if (stmt->IsInstance<tir::AllocateConstNode>()) {
      buffer_info_constants.push_back(kv.first);
    } else {
      buffer_info_vars.push_back(kv.first);
    }
  }
  ICHECK(pass_data_.buffer_info_map_.size() ==
         buffer_info_vars.size() + buffer_info_constants.size())
      << "missing value";

  Map<ObjectRef, ObjectRef> srch;
  // Then intersect constants with each other, as all constants should exist at the same time:
  for (const auto& buf : buffer_info_constants) {
    srch.Set(buf, buf);
    Array<ObjectRef> conflicts;
    std::copy_if(buffer_info_constants.begin(), buffer_info_constants.end(),
                 std::back_inserter(conflicts), [buf](const auto& b) { return b != buf; });
    buf->conflicts.Assign(conflicts.begin(), conflicts.end());
  }

  // And third, remove all conflicts between constants and vars:
  for (const auto& buf : buffer_info_vars) {
    Array<ObjectRef> conflicts;
    std::copy_if(buf->conflicts.begin(), buf->conflicts.end(), std::back_inserter(conflicts),
                 [&srch](const auto& c) { return srch.end() == srch.find(c); });
    buf->conflicts.Assign(conflicts.begin(), conflicts.end());
  }
  return BufferInfoAnalysis(this->pass_data_.buffer_info_map_, max_open_set_size);
}

tir::usmp::BufferInfoAnalysis ExtractBufferInfo(const relax::Function& main_func,
                                                const IRModule& mod) {
  return BufferInfoExtractor(mod)(main_func);
}

TVM_REGISTER_GLOBAL("relax.analysis.extract_buffer_info")
    .set_body_typed([](relax::Function main_func, IRModule mod) {
      return (ExtractBufferInfo(main_func, mod));
    });

}  // namespace tvm
