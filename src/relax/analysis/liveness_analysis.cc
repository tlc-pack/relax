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

#include "liveness_analysis.h"

#include <list>
#include <utility>

namespace tvm {
namespace relax {

using support::Arena;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

ControlFlowGraph ControlFlowGraph::Create(Arena* arena, const Expr& body) {
  return Creator().Create(arena, body);
}

ControlFlowGraph ControlFlowGraph::Creator::Create(Arena* arena, const Expr& body) {
  arena_ = arena;
  cfg_.entry = BasicBlock::Make(arena);
  parent = cfg_.entry;
  VisitExpr(body);
  return std::move(cfg_);
}

void ControlFlowGraph::Creator::Succ(BasicBlockPtr from, BasicBlockPtr to) {
  from->succ.push_back(to);
  to->pred.push_back(from);
}

void ControlFlowGraph::Creator::VisitExpr_(const FunctionNode* f) {
  ICHECK(!in_func_) << "nested functions not supported by CFG analysis";
  in_func_ = true;
  return VisitExpr(f->body);
}

void ControlFlowGraph::Creator::VisitBinding_(const VarBindingNode* var_binding_node) {
  auto var_binding = GetRef<VarBinding>(var_binding_node);

  NodePtr curr_node = Node::Make(arena_, parent, var_binding);

  ICHECK(!cfg_.var_binding_map.count(var_binding));
  cfg_.var_binding_map[var_binding] = curr_node;
  cfg_.reverse_post_order.push_back(curr_node);

  // The basic block ends upon reaching control flow, with successor blocks corresponding to the
  // control flow branch exprs (true/false in If).
  if (const IfNode* ite = var_binding_node->value.as<IfNode>()) {
    // Create the basic blocks for each branch and mark them as successors to the current block.
    BasicBlockPtr t_block = BasicBlock::Make(arena_);
    BasicBlockPtr f_block = BasicBlock::Make(arena_);
    Succ(parent, t_block);
    Succ(parent, f_block);

    BasicBlockPtr old_parent = parent;
    parent = t_block;
    VisitExpr(ite->true_branch);
    parent = f_block;
    VisitExpr(ite->false_branch);
    parent = old_parent;

    // All subsequent bindings (and/or the body expr) will be in a new basic block.
    BasicBlockPtr next = BasicBlock::Make(arena_);
    Succ(t_block, next);
    Succ(f_block, next);
    parent = next;

    return;
  } else if (const CallNode* call_node = var_binding_node->value.as<CallNode>()) {
    if (call_node->op.as<ExternFuncNode>()) {
      // This is a call to an extern func so register the var it binds to.
      cfg_.call_packed_returns.insert(var_binding_node->var);
    }
    // No need to visit the call node and add it to the reverse_post_order since that will
    // add an extra use of the call parameters after the binding itself.
    return;
  }
  ExprVisitor::VisitBinding_(var_binding_node);
}

void ControlFlowGraph::Creator::VisitExpr_(const IfNode* if_node) {
  LOG(FATAL) << "If expressions should be bound to variables.";
}

void ControlFlowGraph::Creator::VisitBinding_(const MatchShapeNode* match_node) {
  // TODO(gigiblender): Handle aliasing.
  auto match_shape = GetRef<MatchShape>(match_node);

  NodePtr curr_node = Node::Make(arena_, parent, match_shape);

  ICHECK(!cfg_.var_binding_map.count(match_shape));
  cfg_.var_binding_map[match_shape] = curr_node;
  cfg_.reverse_post_order.push_back(curr_node);

  // The basic block ends upon reaching control flow, with successor blocks corresponding to the
  // control flow branch exprs (true/false in If).
  if (const IfNode* ite = match_node->value.as<IfNode>()) {
    // Create the basic blocks for each branch and mark them as successors to the current block.
    BasicBlockPtr t_block = BasicBlock::Make(arena_);
    BasicBlockPtr f_block = BasicBlock::Make(arena_);
    Succ(parent, t_block);
    Succ(parent, f_block);

    BasicBlockPtr old_parent = parent;
    parent = t_block;
    VisitExpr(ite->true_branch);
    parent = f_block;
    VisitExpr(ite->false_branch);
    parent = old_parent;

    // All subsequent bindings (and/or the body expr) will be in a new basic block.
    BasicBlockPtr next = BasicBlock::Make(arena_);
    Succ(t_block, next);
    Succ(f_block, next);
    parent = next;
    return;
  } else if (const CallNode* call_node = match_node->value.as<CallNode>()) {
    if (call_node->op.as<ExternFuncNode>()) {
      // This is a call to an extern func so register the var it binds to.
      cfg_.call_packed_returns.insert(match_node->var);
    }
    // No need to visit the call node and add it to the reverse_post_order since that will
    // add an extra use of the call parameters after the binding itself.
    return;
  }
  ExprVisitor::VisitBinding_(match_node);
}

void VarUseCollector::VisitExpr_(const VarNode* var_node) { use_set.insert(GetRef<Var>(var_node)); }

void VarUseCollector::VisitExpr_(const CallNode* call_node) {
  VisitExpr(call_node->op);
  for (const Expr& arg : call_node->args) {
    VisitExpr(arg);
  }
}

void VarUseCollector::VisitExpr_(const TupleNode* tuple_node) {
  for (const Expr& field : tuple_node->fields) {
    VisitExpr(field);
  }
}

void VarUseCollector::VisitExpr_(const TupleGetItemNode* get_node) { VisitExpr(get_node->tuple); }

void VarUseCollector::VisitExpr_(const IfNode* if_node) { VisitExpr(if_node->cond); }

VarSet UseDefAnalysis::GetUse(Expr expr) {
  use_collector.use_set = VarSet();
  use_collector.VisitExpr(expr);
  return use_collector.use_set;
}

UseDefAnalysis UseDefAnalysis::Analyze(const CFG& cfg) {
  UseDefAnalysis a;

  // One pass is sufficient.
  for (auto it = cfg.reverse_post_order.begin(); it != cfg.reverse_post_order.end(); ++it) {
    const CFG::NodePtr& node = *it;
    if (const VarBindingNode* binding_node = node->binding.as<VarBindingNode>()) {
      a.use[node] = a.GetUse(binding_node->value);
      a.def[node] = binding_node->var;
    } else if (const MatchShapeNode* match_shape_node = node->binding.as<MatchShapeNode>()) {
      // TODO(gigiblender): Handle aliasing.
      a.use[node] = a.GetUse(match_shape_node->value);
      a.def[node] = match_shape_node->var;
    } else if (const ExprNode* expr_node = node->binding.as<ExprNode>()) {
      a.use[node] = a.GetUse(runtime::GetRef<Expr>(expr_node));
      a.def[node] = Var("empty_var", NullOpt, NullOpt, Span());
    }
  }
  return a;
}

bool SetEqual(const VarSet& a, const VarSet& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto& xa : a) {
    if (!b.count(xa)) {
      return false;
    }
  }
  return true;
}

LivenessAnalysis LivenessAnalysis::Analyze(const ControlFlowGraph& cfg,
                                           const UseDefAnalysis& use_def) {
  LivenessAnalysis a;
  std::list<CFG::NodePtr> worklist;

  // Initialize worklist to post-order traversal for quick convergence.
  worklist.insert(worklist.end(), cfg.reverse_post_order.rbegin(), cfg.reverse_post_order.rend());

  // See https://lambda.uta.edu/cse5317/notes/node40.html for an overview of the algorithm.
  auto visitor = [&](const CFG::NodePtr n) {
    VarSet old_in_n = a.live_in[n];
    VarSet old_out_n = a.live_out[n];

    a.live_in[n] = use_def.use.at(n);
    for (const Var& v : a.live_out[n]) {
      if (!v.same_as(use_def.def.at(n))) {
        a.live_in[n].insert(v);
      }
    }

    a.live_out[n] = VarSet();
    for (const CFG::NodePtr& s : n->GetSucc()) {
      a.live_out[n].insert(a.live_in[s].begin(), a.live_in[s].end());
    }

    if (SetEqual(old_in_n, a.live_in[n]) && SetEqual(old_out_n, a.live_out[n])) {
      // No need to update the worklist.
    } else {
      // Add predecessor nodes back to worklist (no need to add successors, since each node's
      // in/out sets are not dependent on its predecessors).
      for (const CFG::NodePtr& p : n->GetPred()) {
        worklist.push_back(p);
      }
    }
  };

  while (!worklist.empty()) {
    const CFG::NodePtr n = worklist.front();
    worklist.pop_front();
    visitor(n);
  }

  return a;
}

}  // namespace relax
}  // namespace tvm
