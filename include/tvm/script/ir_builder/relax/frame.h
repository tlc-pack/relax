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
#ifndef TVM_SCRIPT_BUILDER_RELAX_FRAME_H_
#define TVM_SCRIPT_BUILDER_RELAX_FRAME_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/ir/frame.h>
#include <tvm/script/ir_builder/ir/ir.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

class RelaxFrameNode : public IRBuilderFrameNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { IRBuilderFrameNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.ir_builder.relax.RelaxFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(RelaxFrameNode, IRBuilderFrameNode);
};

class RelaxFrame : public IRBuilderFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RelaxFrame, IRBuilderFrame, RelaxFrameNode);

 protected:
  RelaxFrame() = default;
};

class BlockFrameNode : public RelaxFrameNode {
 public:
  bool is_dataflow;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("is_dataflow", &is_dataflow);
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.BlockFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockFrameNode, RelaxFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class BlockFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockFrame, RelaxFrame, BlockFrameNode);
};

class FunctionFrameNode : public RelaxFrameNode {
 public:
  Optional<String> name;
  Array<tvm::relax::Var> params;
  Optional<Type> ret_type;
  Map<String, ObjectRef> attrs;
  Array<tvm::relax::BindingBlock> binding_blocks;
  Array<tvm::relax::Expr> outputs;
  tvm::relax::BlockBuilder block_builder;
  BlockFrame default_binding_block_frame{nullptr};

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("params", &params);
    v->Visit("ret_type", &ret_type);
    v->Visit("attrs", &attrs);
    v->Visit("binding_blocks", &binding_blocks);
    v->Visit("outputs", &outputs);
    // `block_builder` is not visited.
    // `default_binding_block_frame` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.relax.FunctionFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionFrameNode, RelaxFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class FunctionFrame : public RelaxFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FunctionFrame, RelaxFrame, FunctionFrameNode);
};

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_BUILDER_RELAX_BASE_H_
