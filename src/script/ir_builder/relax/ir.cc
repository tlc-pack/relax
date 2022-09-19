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
#include <tvm/script/ir_builder/relax/ir.h>

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
      VarNode* var = const_cast<VarNode*>(node.as<VarNode>());
      var->vid = tvm::relax::Id(name);
    });

TVM_STATIC_IR_FUNCTOR(Namer, vtable)
    .set_dispatch<tvm::relax::DataflowVarNode>([](const ObjectRef& node, String name) -> void {
      using tvm::relax::DataflowVarNode;
      DataflowVarNode* var = const_cast<DataflowVarNode*>(node.as<DataflowVarNode>());
      var->vid = tvm::relax::Id(name);
    });

////////////////////////////// Tensor Type //////////////////////////////

TensorType::TensorType(tvm::relax::DynTensorType type, Optional<tvm::relax::Expr> shape) {
  auto n = make_object<TensorTypeNode>();
  n->type = std::move(type);
  n->shape = std::move(shape);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorTypeNode);

TensorType Tensor(Optional<Array<PrimExpr>> shape, DataType dtype, int ndim) {
  using namespace tvm::relax;
  if (shape.defined() && ndim >= 0) {
    CHECK_EQ(shape.value().size(), ndim)
        << "The dimension of the given shape is mismatched with the given `ndim`";
  } else if (shape.defined()) {
    ndim = shape.value().size();
  }
  Optional<Expr> shape_expr = NullOpt;
  if (shape.defined()) {
    shape_expr = ShapeExpr(shape.value());
  }
  return TensorType(DynTensorType(ndim, dtype), shape_expr);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Tensor").set_body_typed(Tensor);

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function() {
  ObjectPtr<FunctionFrameNode> n = make_object<FunctionFrameNode>();
  n->block_builder = tvm::relax::BlockBuilder::Create(/*mod=*/NullOpt);
  return FunctionFrame(n);
}

tvm::relax::Var Arg(const String& name, const tvm::relax::ShapeExpr& shape, const Type& type) {
  FunctionFrame frame = FindFunctionFrame("R.Arg");
  tvm::relax::Var var(name, shape, type);
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

void FuncRetType(tvm::Type ret_type) {
  FunctionFrame frame = FindFunctionFrame("R.ret_type");
  if (frame->ret_type.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function return type, previous one is:\n "
               << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
}

void FuncRetValue(const tvm::relax::Expr& value) {
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
  frame->output = value;
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Function").set_body_typed(Function);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Arg").set_body_typed(Arg);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetType").set_body_typed(FuncRetType);
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
  Array<tvm::relax::Var> emitted_vars = block_frame.value()->emitted_vars;
  for (const tvm::relax::Var& var : vars) {
    CHECK(!var->IsInstance<tvm::relax::DataflowVarNode>())
        << "ValueError: The output variables of a dataflow block must be all global variables.";
    CHECK(std::find(emitted_vars.begin(), emitted_vars.end(), var) != emitted_vars.end())
        << "ValueError: An output variable is not emitted by this dataflow block. Please make sure "
           "all dataflow block output variables are emitted exactly by this block.";
  }

  // Step 4. All normal variables emitted by this dataflow blocks should be output variables.
  for (const tvm::relax::Var& emitted_var : emitted_vars) {
    if (!emitted_var->IsInstance<tvm::relax::DataflowVarNode>()) {
      CHECK(std::find(vars.begin(), vars.end(), emitted_var) != vars.end())
          << "ValueError: An non-dataflow variable of this dataflow block is not an output "
             "variable. Please make sure all non-dataflow variables emitted by this block are all "
             "contained in the output variable list.";
    }
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Dataflow").set_body_typed(Dataflow);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.BindingBlock").set_body_typed(BindingBlock);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.DataflowBlockOutput")
    .set_body_typed(DataflowBlockOutput);

/////////////////////////////// Bindings ///////////////////////////////

tvm::relax::Var Emit(const tvm::relax::Expr& expr, bool is_dataflow_var) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();
  tvm::relax::Var var{nullptr};
  if (block_frame->is_dataflow && !is_dataflow_var) {
    var = block_builder->EmitOutput(expr);
  } else {
    var = block_builder->Emit(expr);
  }
  block_frame->emitted_vars.push_back(var);
  return var;
}

TVM_DLL Optional<tvm::relax::Var> EmitMatchShape(const tvm::relax::Expr& value,   //
                                                 const Array<PrimExpr>& pattern,  //
                                                 bool emit_var,                   //
                                                 bool is_dataflow_var) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  // If we don't intend to emit a variable, just emit the binding and return.
  if (!emit_var) {
    tvm::relax::MatchShape match_shape(value, pattern, tvm::relax::Var{nullptr});
    block_builder->EmitMatchShape(match_shape);
    return NullOpt;
  }

  // TODO(tvm-team): Enhance the API of EmitMatchShape in BlockBuilder and then update the following
  // code snippet
  tvm::relax::Var var{nullptr};
  tvm::relax::Id vid(is_dataflow_var ? "lv" : "gv");

  if (is_dataflow_var) {
    var = tvm::relax::DataflowVar(vid, NullOpt, NullOpt);
  } else {
    var = tvm::relax::Var(vid, NullOpt, NullOpt);
  }

  if (value->checked_type().as<tvm::relax::ShapeTypeNode>()) {
    UpdateType(var, tvm::relax::ShapeType());
  } else if (const tvm::relax::DynTensorTypeNode* tty =
                 value->checked_type().as<tvm::relax::DynTensorTypeNode>()) {
    tvm::relax::ShapeExpr shape = tvm::relax::ShapeExpr(pattern);
    UpdateShape(var, shape);
    DataType dtype = tty->dtype;
    UpdateType(var, tvm::relax::DynTensorType(pattern.size(), dtype));
  } else {
    LOG(FATAL) << "The value passed to EmitMatchShape must be of DynTensorType or ShapeType.";
  }

  block_frame->emitted_vars.push_back(var);
  return block_builder->EmitMatchShape(tvm::relax::MatchShape(value, pattern, var));
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Emit").set_body_typed(Emit);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.EmitMatchShape").set_body_typed(EmitMatchShape);

///////////////////////////// Type Deduce //////////////////////////////

void AnnotateTypeShape(const tvm::relax::Var& var, const Type& type,
                       const Optional<tvm::relax::ShapeExpr>& shape) {
  using tvm::relax::IsBaseOf;
  if (!var->checked_type_.defined()) {
    var->checked_type_ = type;
  } else {
    const Type& var_type = var->checked_type();
    if (IsBaseOf(var_type, type)) {
      var->checked_type_ = type;
    } else if (IsBaseOf(type, var_type)) {
      // The var type is more detailed, do nothing.
    } else {
      LOG(FATAL) << "TypeError: The var type and the annotated type are not competitive.";
    }
  }

  if (!var->shape_.defined()) {
    var->shape_ = shape;
  } else if (shape.defined()) {
    const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
    tvm::relax::Expr var_shape = Downcast<tvm::relax::Expr>(var->shape_.value());
    block_builder->CanProveShapeEqual(var_shape, shape.value());
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.AnnotateTypeShape").set_body_typed(AnnotateTypeShape);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
