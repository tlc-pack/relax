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
#ifndef TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_

#include <tvm/relax/block_builder.h>
#include <tvm/relax/expr.h>
#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/ir/frame.h>
#include <tvm/script/ir_builder/ir/ir.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

/*! \brief The base ir_builder frame for the relax dialect. */
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

/*! \brief The ir_builder frame for the relax function. */
class FunctionFrameNode : public RelaxFrameNode {
 public:
  /*!
   * \brief The function name.
   * \note The name will not be specified in constructor, so it is "Optional",
   *       However, we must specify the name by `R.func_name` before exit this frame.
   */
  Optional<String> name;
  /*! \brief The function params. */
  Array<tvm::relax::Var> params;
  /*!
   * \brief The function return type.
   * \note Usually the function return type can be deduced by the function body.
   *       But we can use this field to specify a more "accurate" return type.
   *       i.e. If the `ret_type` is None, try to use the deduced type from body
   *       If the `ret_type` is not None, check the deduced type is a base type of the given one.
   */
  Optional<Type> ret_type;
  /*! \brief The function attributes. */
  Map<String, ObjectRef> attrs;
  /*! \brief The binding blocks inside the function. */
  Array<tvm::relax::BindingBlock> binding_blocks;
  /*! \brief The function output expr. `NullOpt` when undefined. */
  Optional<tvm::relax::Expr> output;
  /*! \brief The block builder to create Relax function. */
  tvm::relax::BlockBuilder block_builder;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("params", &params);
    v->Visit("ret_type", &ret_type);
    v->Visit("attrs", &attrs);
    v->Visit("binding_blocks", &binding_blocks);
    v->Visit("output", &output);
    // `block_builder` is not visited.
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

/*! \brief The ir_builder frame for relax binding blocks. */
class BlockFrameNode : public RelaxFrameNode {
 public:
  /*! \brief The flag that indicates whether the block is a dataflow block. */
  bool is_dataflow;

  // The following fields are only be used when this frame is a dataflow block frame.
  /*! \brief The names of the global variables in a dataflow block. */
  Optional<Array<String>> output_var_names;
  /*!
   * \brief A boolean indicating if the dataflow block is ended of construction. If it is true, any
   * new binding trying to be emitted into this block will cause an error.
   */
  bool block_ended;
  /*!
   * \brief A name table used to get unique variable names when constructing a dataflow block.
   * \details Since dataflow block will be visited twice during construction, in order to keep the
   * new variable names consistent in both visits, we keep a copy of the block builder's name table
   * when the block frame is being initialized. In the first visit of a dataflow block, we use the
   * block builder's internal name table to get unique variable names. In the second visit, we use
   * this name table for the same purpose. Since in both visits the bindings being emitted are
   * always the same, the new variable names will be consistent with this copy of name table.
   */
  tvm::relax::NameTable name_table;

  void VisitAttrs(tvm::AttrVisitor* v) {
    RelaxFrameNode::VisitAttrs(v);
    v->Visit("is_dataflow", &is_dataflow);
    v->Visit("output_var_names", &output_var_names);
    // `block_ended` is not visited
    // `name_table` is not visited
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

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_RELAX_FRAME_H_
