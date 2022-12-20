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
#include <tvm/relax/analysis.h>
#include <tvm/relax/type_analysis.h>
#include <tvm/script/ir_builder/relax/ir.h>
#include <tvm/tir/op.h>

#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace relax {

///////////////////////////////// Vars //////////////////////////////////

using tvm::script::ir_builder::details::Namer;

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::VarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::VarNode;
      const VarNode* var = node.as<VarNode>();
      relay::IdNode* vid = const_cast<relay::IdNode*>(var->vid.get());
      vid->name_hint = name;
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::DataflowVarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::DataflowVarNode;
      const DataflowVarNode* var = node.as<DataflowVarNode>();
      relay::IdNode* vid = const_cast<relay::IdNode*>(var->vid.get());
      vid->name_hint = name;
    });

////////////////////////////// Tensor Type //////////////////////////////

using tvm::relax::TensorStructInfo;
using tvm::relax::TupleStructInfo;

TensorStructInfo Tensor(Optional<Array<PrimExpr>> shape, DataType dtype, int ndim) {
  using namespace tvm::relax;
  ICHECK_GE(ndim, -1) << "ndim must be >= -1, but got " << ndim;
  if (shape.defined() && ndim >= 0) {
    CHECK_EQ(shape.value().size(), ndim)
        << "The dimension of the given shape is mismatched with the given `ndim`";
  } else if (shape.defined()) {
    ndim = shape.value().size();
  }
  if (shape.defined()) {
    ShapeExpr shape_expr(shape.value());
    return TensorStructInfo(shape_expr, dtype);
  } else {
    return TensorStructInfo(dtype, ndim);
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Tensor").set_body_typed(Tensor);

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function() {
  ObjectPtr<FunctionFrameNode> n = make_object<FunctionFrameNode>();
  const IRBuilder& ir_builder = IRBuilder::Current();
  Optional<tvm::IRModule> mod = NullOpt;
  if (const Optional<ir::IRModuleFrame> mod_frame = ir_builder->GetLastFrame<ir::IRModuleFrame>()) {
    mod = tvm::IRModule(mod_frame.value()->functions);
  }
  n->block_builder = tvm::relax::BlockBuilder::Create(/*mod=*/mod);
  return FunctionFrame(n);
}

tvm::relax::Var Arg(const String& name, const tvm::relax::StructInfo& struct_info) {
  FunctionFrame frame = FindFunctionFrame("R.Arg");
  tvm::relax::Var var(name, struct_info);
  frame->params.push_back(var);
  return var;
}

void FuncName(const String& name) {
  FunctionFrame frame = FindFunctionFrame("R.func_name");
  if (frame->name.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function name, previous one is: \"" << frame->name.value()
               << "\"";
  }
  frame->name = name;
}

void FuncAttrs(Map<String, ObjectRef> attrs) {
  FunctionFrame frame = FindFunctionFrame("R.func_attr");
  if (!frame->attrs.empty()) {
    LOG(FATAL) << "ValueError: Duplicate function attrs, previous one is:\n" << frame->attrs;
  }
  frame->attrs = attrs;
}

void FuncRetStructInfo(const tvm::relax::StructInfo& ret_sinfo) {
  FunctionFrame frame = FindFunctionFrame("R.func_ret_struct_info");
  if (frame->ret_struct_info.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function return struct info, previous one is:\n "
               << frame->ret_struct_info.value();
  }
  frame->ret_struct_info = ret_sinfo;
}

void FuncRetValue(const tvm::relax::Expr& value) {
  // Step 0. Normalize the value.
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::Expr normalized_value = block_builder->Normalize(value);

  // Step 1. The current Relax TVMScript syntax only allows function return appearing at the end of
  // a function body. Therefore if there is any unended block frame when dealing with function
  // return, we should end the block frame.
  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  if (block_frame.defined()) {
    block_frame.value()->ExitWithScope();
    ICHECK(!IRBuilder::Current()->FindFrame<BlockFrame>())
        << "All block frame are supposed to be popped out already";
  }
  // Step 2. Add the output value to the function frame.
  FunctionFrame frame = FindFunctionFrame("return");
  CHECK(!frame->output.defined())
      << "ValueError: Relax functions don't support multiple return statement. Please make sure "
         "the return statement appears at the end of function.";

  frame->output = std::move(normalized_value);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Function").set_body_typed(Function);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Arg").set_body_typed(Arg);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetStructInfo").set_body_typed(FuncRetStructInfo);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetValue").set_body_typed(FuncRetValue);

///////////////////////////// BindingBlock //////////////////////////////

BlockFrame Dataflow() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = true;
  n->block_ended = false;
  return BlockFrame(n);
}

BlockFrame BindingBlock() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = false;
  n->block_ended = false;
  return BlockFrame(n);
}

void DataflowBlockOutput(const Array<tvm::relax::Var>& vars) {
  // Step 1. Check that we're in a Dataflow block that is not ended.
  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  CHECK(block_frame.defined() && block_frame.value()->is_dataflow)
      << "ValueError: `R.output` should appear inside a dataflow block. However, the current "
         "innermost block is not a dataflow block.";
  CHECK(!block_frame.value()->block_ended)
      << "ValueError: It is not allowed for a dataflow block to have multiple output operation.";

  // Step 2. Mark the block frame ended of construction, so that any followup binding after this
  // mark in the dataflow block will lead to an error.
  block_frame.value()->block_ended = true;

  // Step 3. All the output variables must be global variables and must be emitted by this dataflow
  // block.
  const Array<tvm::relax::Var>& emitted_vars = block_frame.value()->emitted_vars;
  for (const tvm::relax::Var& var : vars) {
    CHECK(std::find(emitted_vars.begin(), emitted_vars.end(), var) != emitted_vars.end())
        << "ValueError: An output variable is not emitted by this dataflow block. Please make sure "
           "all dataflow block output variables are emitted exactly by this block.";
    block_frame.value()->output_vars.push_back(var);
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Dataflow").set_body_typed(Dataflow);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.BindingBlock").set_body_typed(BindingBlock);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.DataflowBlockOutput")
    .set_body_typed(DataflowBlockOutput);

/////////////////////////////// Bindings ///////////////////////////////

tvm::relax::Var Emit(const tvm::relax::Expr& expr) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
  tvm::relax::Var var{nullptr};
  var = block_builder->Emit(expr);
  block_frame->emitted_vars.push_back(var);
  return var;
}

Optional<tvm::relax::Var> EmitMatchShape(const tvm::relax::Expr& value,   //
                                         const Array<PrimExpr>& pattern,  //
                                         bool emit_var) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  if (!emit_var) {
    // If we don't intend to emit a variable, just emit the binding and return.
    tvm::relax::MatchShape match_shape(value, pattern, tvm::relax::Var{nullptr});
    block_builder->EmitMatchShape(match_shape);
    return NullOpt;
  } else {
    // Otherwise, we need to emit a variable and bind it to the match shape.
    tvm::relax::Var var = block_builder->EmitMatchShape(value, pattern);
    block_frame->emitted_vars.push_back(var);
    return var;
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Emit").set_body_typed(Emit);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.EmitMatchShape").set_body_typed(EmitMatchShape);

///////////////////////////// Type Deduce //////////////////////////////

void AnnotateStructInfo(const tvm::relax::Var& var,
                        const tvm::relax::StructInfo& anno_struct_info) {
  using tvm::relax::IsBaseOf;
  using tvm::relax::StructInfo;

  Type type = GetStaticType(anno_struct_info);
  Optional<tvm::relax::Expr> shape = GetLegacyShapeHint(anno_struct_info);

  // TODO(siyuan, ruihang): Revisit the checks aftr we fully migrate to struct info.
  // consider simplify assumption to require var not contain struct info at all.
  if (var->struct_info_.defined()) {
    // Case 1. The var already has struct info.
    const StructInfo& var_struct_info = Downcast<StructInfo>(var->struct_info_.value());
    CHECK(IsBaseOf(var_struct_info, anno_struct_info) ||
          IsBaseOf(anno_struct_info, var_struct_info))
        << "ValueError: The annotation struct info is not a base of the existing struct info.";
  } else {
    // Case 2. The var doesn't have struct info.
    // This path may be removed later

    if (var->checked_type_.defined()) {
      const Type& var_type = var->checked_type();
      CHECK(IsBaseOf(type, var_type) || IsBaseOf(var_type, type))
          << "TypeError: The annotated type and value type are not compatible. "
          << "The Type is expected to be " << var_type << " but got annotation: " << type;
    }
    if (var->shape_.defined() && shape.defined()) {
      tvm::relax::Expr var_shape = Downcast<tvm::relax::Expr>(var->shape_.value());
      auto check_shape = [](const tvm::relax::Expr& lhs, const tvm::relax::Expr& rhs) {
        if (lhs->IsInstance<tvm::relax::RuntimeDepShapeNode>() ||
            rhs->IsInstance<tvm::relax::RuntimeDepShapeNode>()) {
          return true;
        } else {
          const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
          return block_builder->CanProveShapeEqual(lhs, rhs);
        }
      };
      CHECK(check_shape(var_shape, shape.value()))
          << " The shape of var " << var->name_hint() << " is expected to be " << var_shape
          << " but got annotation: " << shape.value();
    }
  }

  var->checked_type_ = type;
  var->shape_ = shape;
  var->struct_info_ = anno_struct_info;
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.AnnotateStructInfo")
    .set_body_typed(AnnotateStructInfo);

///////////////////////////// If Then Else /////////////////////////////

IfFrame If(tvm::relax::Expr condition) {
  ObjectPtr<IfFrameNode> n = make_object<IfFrameNode>();
  n->condition = condition;
  n->then_expr = NullOpt;
  n->else_expr = NullOpt;
  return IfFrame(n);
}

ThenFrame Then() {
  ObjectPtr<ThenFrameNode> n = make_object<ThenFrameNode>();
  return ThenFrame(n);
}

ElseFrame Else() {
  ObjectPtr<ElseFrameNode> n = make_object<ElseFrameNode>();
  return ElseFrame(n);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.If").set_body_typed(If);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Then").set_body_typed(Then);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Else").set_body_typed(Else);

//////////////////////// Symbolic Shape Rewriter ////////////////////////

using tvm::relax::Expr;
class SymbolicShapeRewriter : public tvm::relax::StructInfoMutator {
 public:
  explicit SymbolicShapeRewriter(Map<String, tvm::tir::Var> var_table)
      : var_table_(std::move(var_table)) {}

  Array<tvm::tir::Var> undefined_vars_;

 private:
  Expr VisitStructInfoExprField(const Expr& expr) {
    if (const auto* shape_expr = expr.as<tvm::relax::ShapeExprNode>()) {
      Array<PrimExpr> new_shape;
      bool changed = false;
      for (const tvm::PrimExpr& s : shape_expr->values) {
        if (const auto* var = s.as<tvm::tir::VarNode>()) {
          auto it = var_table_.find(var->name_hint);
          if (it != var_table_.end()) {
            new_shape.push_back((*it).second);
            changed = true;
          } else {
            undefined_vars_.push_back(GetRef<tvm::tir::Var>(var));
            var_table_.Set(var->name_hint, GetRef<tvm::tir::Var>(var));
            new_shape.push_back(s);
          }
        } else {
          // TODO(siyuan, ruihang): confirm and use VisitPrimExpr to recursive rewrite.
          new_shape.push_back(s);
        }
      }
      if (changed) {
        return tvm::relax::ShapeExpr(new_shape);
      }
    }
    return expr;
  }

 private:
  Map<String, tvm::tir::Var> var_table_;
};

TVM_REGISTER_GLOBAL("script.ir_builder.relax.RewriteSymbolicShape")
    .set_body_typed([](tvm::relax::StructInfo struct_info, Map<String, tvm::tir::Var> var_table) {
      SymbolicShapeRewriter rewriter(var_table);
      tvm::relax::StructInfo rewritten_info = rewriter(std::move(struct_info));
      return Array<ObjectRef>{rewritten_info, rewriter.undefined_vars_};
    });

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
