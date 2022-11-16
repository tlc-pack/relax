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

#ifndef TVM_RELAX_ANALYSIS_LIVENESS_ANALYSIS_H_
#define TVM_RELAX_ANALYSIS_LIVENESS_ANALYSIS_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../support/arena.h"
#include "tvm/relax/expr.h"
#include "tvm/relax/expr_functor.h"

namespace tvm {
namespace relax {

using support::Arena;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief A representation of an input expression (typically a Function) as a directed graph of
 * basic blocks, with edges between basic blocks corresponding to control flow branching.
 */
class ControlFlowGraph {
 public:
  struct Node;
  struct BasicBlock;

  using NodePtr = Node*;
  using BasicBlockPtr = BasicBlock*;

  /*!
   * \brief A chunk of IR that does not have any control flow branching.
   */
  struct BasicBlock {
    // The nodes of the basic block.
    std::vector<NodePtr> nodes;
    // The predecessor basic blocks.
    std::vector<BasicBlockPtr> pred;
    // The successor basic blocks.
    std::vector<BasicBlockPtr> succ;

    static BasicBlockPtr Make(support::Arena* arena) { return arena->make<BasicBlock>(); }
  };

  /*!
   * \brief Roughly corresponds to a "statement" in the IR, such as an individual binding in a
   * basic block or the "return value" of a block. Each node maps to a single corresponding expr in
   * the IR, but the converse is not true (e.g. in the case of variables).
   */
  struct Node {
    /*! \brief The basic block this node belongs to. */
    BasicBlockPtr parent;
    /*! \brief The index into the parent basic block where this node is. */
    size_t index;
    /*! \brief The binding this node corresponds to. */
    runtime::ObjectRef binding;

    /*! \brief Returns whether or not this node is the first one in the parent basic block. */
    bool IsFirst() const { return index == 0; }

    /*! \brief Returns whether or not this node is the last one in the parent basic block. */
    bool IsLast() const { return index == parent->nodes.size() - 1; }

    /*! \brief Returns the predecessor nodes of this node. */
    std::vector<NodePtr> GetPred() const {
      std::vector<NodePtr> pred;
      if (IsFirst()) {
        for (const BasicBlockPtr& pred_block : parent->pred) {
          if (!pred_block->nodes.empty()) {
            pred.insert(pred.end(), pred_block->nodes.rbegin(), pred_block->nodes.rend());
          }
        }
      } else {
        pred.insert(pred.end(), parent->nodes.rbegin() + index + 1, parent->nodes.rend());
      }
      return pred;
    }

    /*! \brief Returns the successor nodes of this node. */
    std::vector<NodePtr> GetSucc() const {
      std::vector<NodePtr> succ;
      if (IsLast()) {
        for (const BasicBlockPtr& succ_block : parent->succ) {
          if (!succ_block->nodes.empty()) {
            succ.insert(succ.end(), succ_block->nodes.begin(), succ_block->nodes.end());
          }
        }
      } else {
        succ.insert(succ.end(), (parent->nodes.begin() + index + 1), parent->nodes.end());
      }
      return succ;
    }

    /*! \brief Creates a node with the given expr/binding and appends it to the
     * parent basic block.
     */
    static NodePtr Make(Arena* arena, BasicBlockPtr parent, runtime::ObjectRef binding) {
      NodePtr n = arena->make<Node>();
      n->parent = parent;
      n->binding = binding;
      n->index = parent->nodes.size();
      parent->nodes.push_back(n);
      return n;
    }
  };

  /*! \brief The basic block where control flow begins. */
  BasicBlockPtr entry;

  /*!
   * \brief Mapping from VarBindings to their corresponding nodes. Note that VarBindings
   * are never shared in ANF (unlike vars), so this is an injection.
   */
  std::unordered_map<runtime::ObjectRef, NodePtr, ObjectPtrHash, ObjectPtrEqual> var_binding_map;

  /*!
   * \brief Set of Vars that are bound to return values of CallPacked functions. We omit killing
   * those.
   */
  VarSet call_packed_returns;

  /*! \brief The nodes of the CFG in reverse post order. */
  std::vector<NodePtr> reverse_post_order;

  /*! \brief Creates and returns the CFG of the given expression. */
  static ControlFlowGraph Create(Arena* arena, const Expr& body);

 private:
  class Creator;
};

/*! \brief Helper class for building CFGs. */
class ControlFlowGraph::Creator : ExprVisitor {
 public:
  Creator() {}

  ControlFlowGraph Create(Arena* arena, const Expr& body);

 private:
  /*! \brief The arena allocator. */
  Arena* arena_;

  /*! \brief The CFG being built. */
  ControlFlowGraph cfg_;

  /*! \brief The current basic block being constructed. */
  BasicBlockPtr parent;

  /*!
   * \brief Whether or not we are in a function. CFGs do not support nested functions so this is
   * used to error out in such a case.
   */
  bool in_func_ = false;

  /*!
   * \brief Link \p to as a successor block to \p from.
   */
  void Succ(BasicBlockPtr from, BasicBlockPtr to);

#define DEFAULT_CFG(OP)                                                     \
  void VisitExpr_(const OP* op) final {                                     \
    NodePtr n = Node::Make(arena_, parent, GetRef<runtime::ObjectRef>(op)); \
    cfg_.reverse_post_order.push_back(n);                                   \
  }

  void VisitExpr_(const FunctionNode* f) final;
  void VisitBinding_(const VarBindingNode* binding_node) final;
  void VisitExpr_(const IfNode* if_node) final;
  void VisitBinding_(const MatchShapeNode* match_node) final;

  DEFAULT_CFG(VarNode);
  DEFAULT_CFG(GlobalVarNode);
  DEFAULT_CFG(ConstantNode);
  DEFAULT_CFG(CallNode);
  DEFAULT_CFG(OpNode);
  DEFAULT_CFG(TupleNode);
  DEFAULT_CFG(TupleGetItemNode);
};

/*!
 * \brief Helper class for collecting the variables used/read by an expression. NOTE: for If exprs,
 * only the condition is included (not the branches).
 */
class VarUseCollector : ExprVisitor {
 public:
  void VisitExpr_(const VarNode* var_node);
  void VisitExpr_(const CallNode* call_node);
  void VisitExpr_(const TupleNode* tuple_node);
  void VisitExpr_(const TupleGetItemNode* get_node);
  void VisitExpr_(const IfNode* if_node);

  void VisitExpr_(const ConstructorNode* cons_node) {}
  void VisitExpr_(const GlobalVarNode* gvar_node) {}
  void VisitExpr_(const ConstantNode* const_node) {}
  void VisitExpr_(const OpNode* op_node) {}
  void VisitExpr_(const FunctionNode* func_node) {}

  void VisitExpr(const Expr& expr) { ExprVisitor::VisitExpr(expr); }

  /*!
   * \brief The current use set.
   */
  VarSet use_set;
};

/*!
 * \brief Analysis that collects the variables used and defined at each node.
 */
struct UseDefAnalysis {
  using CFG = ControlFlowGraph;

  /*! \brief Mapping of node -> variables used/read by node. */
  std::unordered_map<CFG::NodePtr, VarSet> use;

  /*! \brief Mapping of node -> variable defined/written by node. */
  std::unordered_map<CFG::NodePtr, Var> def;

  VarUseCollector use_collector;

  VarSet GetUse(Expr expr);

  static UseDefAnalysis Analyze(const CFG& cfg);
};

/*! \brief Returns whether \p a and \p b are the same set of vars. */
bool SetEqual(const VarSet& a, const VarSet& b);

/*!
 * \brief Analysis that collects the live variables before and after each node.
 */
struct LivenessAnalysis {
  using CFG = ControlFlowGraph;

  /*! \brief Mapping of node -> set of variables live before node. */
  std::unordered_map<CFG::NodePtr, VarSet> live_in;

  /*! \brief Mapping of node -> set of variables live after node. */
  std::unordered_map<CFG::NodePtr, VarSet> live_out;

  /*!
   * \brief Analyze the input \p cfg (using info from \p use_def).
   *
   * \param cfg The input control flow graph.
   * \param use_def Use-def analysis of \p cfg.
   * \return LivenessAnalysis
   */
  static LivenessAnalysis Analyze(const ControlFlowGraph& cfg, const UseDefAnalysis& use_def);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_ANALYSIS_LIVENESS_ANALYSIS_H_
