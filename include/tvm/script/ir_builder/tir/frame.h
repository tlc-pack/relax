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
#ifndef TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_
#define TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_

#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/ir/frame.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

class TIRFrameNode : public IRBuilderFrameNode {
 public:
  Array<tvm::tir::Stmt> stmts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    IRBuilderFrameNode::VisitAttrs(v);
    v->Visit("stmts", &stmts);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.TIRFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(TIRFrameNode, IRBuilderFrameNode);
};

class TIRFrame : public IRBuilderFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TIRFrame, IRBuilderFrame, TIRFrameNode);

 protected:
  TIRFrame() = default;
};

class BlockFrameNode : public TIRFrameNode {
 public:
  String name;
  Array<tvm::tir::IterVar> iter_vars;
  Optional<Array<tvm::tir::BufferRegion>> reads;
  Optional<Array<tvm::tir::BufferRegion>> writes;
  Optional<tvm::tir::Stmt> init;
  Array<tvm::tir::Buffer> alloc_buffers;
  Array<tvm::tir::MatchBufferRegion> match_buffers;
  Optional<Map<String, ObjectRef>> annotations;

  Array<PrimExpr> iter_values;
  Optional<PrimExpr> predicate;
  bool no_realize;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("iter_vars", &iter_vars);
    v->Visit("reads", &reads);
    v->Visit("writes", &writes);
    v->Visit("init", &init);
    v->Visit("alloc_buffers", &alloc_buffers);
    v->Visit("match_buffers", &match_buffers);
    v->Visit("annotations", &annotations);
    v->Visit("iter_values", &iter_values);
    v->Visit("predicate", &predicate);
    v->Visit("no_realize", &no_realize);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.BlockFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class BlockFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockFrame, TIRFrame, BlockFrameNode);
};

class BlockInitFrameNode : public TIRFrameNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) { TIRFrameNode::VisitAttrs(v); }

  static constexpr const char* _type_key = "script.ir_builder.tir.BlockInitFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(BlockInitFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class BlockInitFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BlockInitFrame, TIRFrame, BlockInitFrameNode);
};

class ForFrameNode : public TIRFrameNode {
 public:
  using FMakeForLoop =
      runtime::TypedPackedFunc<tvm::tir::Stmt(Array<tvm::tir::Var>, Array<Range>, tvm::tir::Stmt)>;

  Array<tvm::tir::Var> vars;
  Array<Range> doms;
  FMakeForLoop f_make_for_loop;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("vars", &vars);
    v->Visit("doms", &doms);
    // `f_make_for_loop` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.ForFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ForFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class ForFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ForFrame, TIRFrame, ForFrameNode);
};

class PrimFuncFrameNode : public TIRFrameNode {
 public:
  Optional<String> name;
  Array<tvm::tir::Var> args;
  Optional<Type> ret_type;
  Map<tvm::tir::Var, tvm::tir::Buffer> buffer_map;
  Map<tvm::tir::Var, tvm::tir::Buffer> preflattened_buffer_map;
  Optional<Map<String, ObjectRef>> attrs;
  Map<tvm::tir::Var, tvm::tir::IterVar> env_threads;
  Array<tvm::tir::Buffer> root_alloc_buffers;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("name", &name);
    v->Visit("args", &args);
    v->Visit("ret_type", &ret_type);
    v->Visit("buffer_map", &buffer_map);
    v->Visit("preflattened_buffer_map", &preflattened_buffer_map);
    v->Visit("attrs", &attrs);
    v->Visit("env_threads", &env_threads);
    v->Visit("root_alloc_buffers", &root_alloc_buffers);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.PrimFuncFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class PrimFuncFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrimFuncFrame, TIRFrame, PrimFuncFrameNode);
};

class AssertFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;
  PrimExpr message;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("message", &message);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AssertFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AssertFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AssertFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AssertFrame, TIRFrame, AssertFrameNode);
};

class LetFrameNode : public TIRFrameNode {
 public:
  tvm::tir::Var var;
  PrimExpr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("var", &var);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.LetFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LetFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class LetFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LetFrame, TIRFrame, LetFrameNode);
};

class AllocateFrameNode : public TIRFrameNode {
 public:
  Array<PrimExpr> extents;
  DataType dtype;
  String storage_scope;
  PrimExpr condition;
  Map<String, ObjectRef> annotations;
  tvm::tir::Buffer buffer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("extents", &extents);
    v->Visit("dtype", &dtype);
    v->Visit("storage_scope", &storage_scope);
    v->Visit("condition", &condition);
    v->Visit("annotations", &annotations);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AllocateFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateFrame, TIRFrame, AllocateFrameNode);
};

class AllocateConstFrameNode : public TIRFrameNode {
 public:
  DataType dtype;
  Array<PrimExpr> extents;
  tvm::runtime::NDArray data;
  tvm::tir::Buffer buffer;
  Map<String, ObjectRef> annotations;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("dtype", &dtype);
    v->Visit("extents", &extents);
    v->Visit("data", &data);
    v->Visit("buffer", &buffer);
    v->Visit("annotations", &annotations);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AllocateConstFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AllocateConstFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AllocateConstFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AllocateConstFrame, TIRFrame,
                                                    AllocateConstFrameNode);
};

class LaunchThreadFrameNode : public TIRFrameNode {
 public:
  PrimExpr extent;
  String attr_key;
  tvm::tir::IterVar iter_var;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("extent", &extent);
    v->Visit("attr_key", &attr_key);
    v->Visit("iter_var", &iter_var);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.LaunchThreadFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(LaunchThreadFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class LaunchThreadFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(LaunchThreadFrame, TIRFrame,
                                                    LaunchThreadFrameNode);
};

class RealizeFrameNode : public TIRFrameNode {
 public:
  tvm::tir::BufferRegion buffer_slice;
  String storage_scope;
  PrimExpr condition;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("buffer_slice", &buffer_slice);
    v->Visit("storage_scope", &storage_scope);
    v->Visit("condition", &condition);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.RealizeFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(RealizeFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class RealizeFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(RealizeFrame, TIRFrame, RealizeFrameNode);
};

class AttrFrameNode : public TIRFrameNode {
 public:
  ObjectRef node;
  String attr_key;
  PrimExpr value;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("node", &node);
    v->Visit("attr_key", &attr_key);
    v->Visit("value", &value);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.AttrFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttrFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class AttrFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(AttrFrame, TIRFrame, AttrFrameNode);
};

class WhileFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.WhileFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(WhileFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class WhileFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(WhileFrame, TIRFrame, WhileFrameNode);
};

class IfFrameNode : public TIRFrameNode {
 public:
  PrimExpr condition;
  Optional<Array<tvm::tir::Stmt>> then_stmts;
  Optional<Array<tvm::tir::Stmt>> else_stmts;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("condition", &condition);
    v->Visit("then_stmts", &then_stmts);
    v->Visit("else_stmts", &else_stmts);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.IfFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(IfFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class IfFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IfFrame, TIRFrame, IfFrameNode);
};

class ThenFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.ir_builder.tir.ThenFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThenFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class ThenFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ThenFrame, TIRFrame, ThenFrameNode);
};

class ElseFrameNode : public TIRFrameNode {
 public:
  static constexpr const char* _type_key = "script.ir_builder.tir.ElseFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(ElseFrameNode, TIRFrameNode);

 public:
  void EnterWithScope() final;
  void ExitWithScope() final;
};

class ElseFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(ElseFrame, TIRFrame, ElseFrameNode);
};

class DeclBufferFrameNode : public TIRFrameNode {
 public:
  tvm::tir::Buffer buffer;

  void VisitAttrs(tvm::AttrVisitor* v) {
    TIRFrameNode::VisitAttrs(v);
    v->Visit("buffer", &buffer);
  }

  static constexpr const char* _type_key = "script.ir_builder.tir.DeclBufferFrame";
  TVM_DECLARE_FINAL_OBJECT_INFO(DeclBufferFrameNode, TIRFrameNode);

 public:
  void ExitWithScope() final;
};

class DeclBufferFrame : public TIRFrame {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(DeclBufferFrame, TIRFrame, DeclBufferFrameNode);
};

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_TIR_FRAME_H_
