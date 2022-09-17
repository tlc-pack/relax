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

tvm::relax::Var Arg(const String& name, const TensorType& type) {
  FunctionFrame frame = FindFunctionFrame("R.Arg");
  tvm::relax::Var var(name, type->shape, type->type);
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
  frame->outputs.push_back(value);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Function").set_body_typed(Function);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Arg").set_body_typed(Arg);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetType").set_body_typed(FuncRetType);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncRetValue").set_body_typed(FuncRetValue);

///////////////////////////// BindingBlock //////////////////////////////

BlockFrame Dataflow() {
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = true;
  n->output_var_names = NullOpt;
  n->block_ended = false;
  n->name_table = tvm::relax::NameTable(*block_builder->name_table());
  return BlockFrame(n);
}

BlockFrame BindingBlock() {
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = false;
  n->output_var_names = NullOpt;
  n->block_ended = false;
  n->name_table = tvm::relax::NameTable(*block_builder->name_table());
  return BlockFrame(n);
}

void DataflowBlockOutput(const Array<tvm::relax::Var>& vars) {
  // The way the parser deals with dataflow block is:
  //   Since we don't know which variables are global variables during variable creation within
  // single round of visit, we adopt a two-round visit to deal with the construction of
  // dataflow block.
  //   - In the first round, all binding variables are created as dataflow variables.
  //   - At the end of the first round, by looking into the arguments of `R.output`, we know and
  //   stores the names of the global variables.
  //   - Then we clear the variable table, as a preparation step for the second round of visit.
  //   - In the second round, we create variables according to their names, by checking whether the
  //   name exists in the stored global variable names.

  // - Therefore, when visiting dataflow output for the first time, we collect the names of the
  // output variables and store them into the current block frame, and then terminate the block
  // being constructed (which is not correct) and start a new block in the block builder.
  // - When visiting for the second time, we mark block construction finished, in order to throw an
  // error when there is any followup binding within scope of this dataflow block.

  // Check that we're in a Dataflow block.
  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  CHECK(block_frame.defined()) << "ValueError: `R.output` should appear inside a dataflow block. "
                                  "However, the current innermost block is not a dataflow block.";

  if (!block_frame.value()->output_var_names.defined()) {
    // Collect the names of the global variables.
    Array<String> output_var_names;
    output_var_names.reserve(vars.size());
    for (const tvm::relax::Var& var : vars) {
      // All the variables must be dataflow variables in the first round. Otherwise, the definition
      // site of this variable must be outside this dataflow block, and we should throw an error.
      CHECK(var->IsInstance<tvm::relax::DataflowVarNode>())
          << "ValueError: Variable " << var
          << " is not defined inside this dataflow block. Please check if it is defined outside "
             "this dataflow block.";
      output_var_names.push_back(var->vid->name_hint);
    }
    block_frame.value()->output_var_names = std::move(output_var_names);

    // End the current block and start a new block.
    const tvm::relax::BlockBuilder& block_builder = GetBlockBuilder();
    block_builder->EndBlock();
    block_builder->BeginDataflowBlock();
  } else {
    // Mark the block frame ended of construction, so that any followup binding after this mark in
    // the dataflow block will lead to an error.
    block_frame.value()->block_ended = true;

    // All the variables must be global variables in the second round. Otherwise, there must be a
    // dataflow block output that appears previously inside this frame, and we should throw an
    // error.
    for (const tvm::relax::Var& var : vars) {
      CHECK(!var->IsInstance<tvm::relax::DataflowVarNode>())
          << "ValueError: One dataflow block only allows one `R.output`. Please check if there is "
             "any previous `R.output` in this dataflow block.";
    }
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Dataflow").set_body_typed(Dataflow);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.BindingBlock").set_body_typed(BindingBlock);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.DataflowBlockOutput")
    .set_body_typed(DataflowBlockOutput);

/////////////////////////////// Bindings ///////////////////////////////

BlockFrame CheckBlockFrameExistAndUnended() {
  // - If we're emitting a non-dataflow binding in the function (that is to say, the binding is not
  // wrapped by `with R.dataflow()`), it is possible that there is no existing BlockFrame. In this
  // case, we will create a BlockFrame and "enter its 'with' scope" first.
  // - Otherwise, there is already an existing BlockFrame. We check if the block is "ended" - if a
  // block is ended, it is not allowed to emit new bindings into this block, and we should throw
  // exceptions.

  Optional<BlockFrame> block_frame = IRBuilder::Current()->GetLastFrame<BlockFrame>();
  if (block_frame.defined()) {
    CHECK(!block_frame.value()->block_ended)
        << "New binding is not allowed after dataflow block output";
    return block_frame.value();
  }

  BlockFrame new_block_frame = BindingBlock();
  new_block_frame->EnterWithScope();
  return new_block_frame;
}

tvm::relax::Var Emit(const tvm::relax::Expr& expr, String var_name) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  // For dataflow blocks,
  // - in the first round of visit, we directly emit the input expression.
  // - in the second round of visit, we create the correct type of variable according to the stored
  // global variable names.
  // For ordinary binding blocks, the expression will only be emitted once in the normal way of
  // emission.

  if (!block_frame->output_var_names.defined()) {
    return block_builder->Emit(expr, var_name);
  } else {
    // If we directly emit the expression by providing the desired variable name, the block builder
    // will automatically rename the variable whatever. Therefore, we manually create the variable
    // with correct fields and the binding. And finally emit the manually created binding.
    var_name = block_frame->name_table.GetUniqueName(var_name);

    tvm::relax::Id vid(var_name);
    tvm::relax::Expr normalized_expr = block_builder->Normalize(expr);

    Array<String> output_var_names = block_frame->output_var_names.value();
    if (std::find(output_var_names.begin(), output_var_names.end(), var_name) !=
        output_var_names.end()) {
      tvm::relax::Var var = tvm::relax::Var(vid, NullOpt, NullOpt);
      UpdateType(var, normalized_expr->checked_type_);
      UpdateShape(var, normalized_expr->shape_);
      return block_builder->EmitOutput(tvm::relax::VarBinding(var, normalized_expr));
    } else {
      tvm::relax::Var var = tvm::relax::DataflowVar(vid, NullOpt, NullOpt);
      UpdateType(var, normalized_expr->checked_type_);
      UpdateShape(var, normalized_expr->shape_);
      return block_builder->Emit(tvm::relax::VarBinding(var, normalized_expr));
    }
  }
}

TVM_DLL Optional<tvm::relax::Var> EmitMatchShape(const tvm::relax::Expr& value,   //
                                                 const Array<PrimExpr>& pattern,  //
                                                 Optional<String> var_name) {
  BlockFrame block_frame = CheckBlockFrameExistAndUnended();
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();

  // If we don't intend to emit a variable, just emit the binding and return.
  if (!var_name.defined()) {
    tvm::relax::MatchShape match_shape(value, pattern, tvm::relax::Var{nullptr});
    block_builder->EmitMatchShape(match_shape);
    return NullOpt;
  }

  // For dataflow blocks,
  // - in the first round of visit, we directly emit the input expression.
  // - in the second round of visit, we create the correct type of variable according to the stored
  // global variable names.
  // For ordinary binding blocks, the expression will only be emitted once in the normal way of
  // emission.

  String var_name_str = var_name.value();
  if (!block_frame->output_var_names.defined()) {
    return block_builder->EmitMatchShape(value, pattern, var_name_str);
  } else {
    // If we directly emit the expression by providing the desired variable name, the block builder
    // will automatically rename the variable whatever. Therefore, we manually create the variable
    // with correct fields and the binding. And finally emit the manually created binding.
    var_name_str = block_frame->name_table.GetUniqueName(var_name_str);

    tvm::relax::Var var{nullptr};
    tvm::relax::Id vid(var_name_str);

    Array<String> output_var_names = block_frame->output_var_names.value();
    if (std::find(output_var_names.begin(), output_var_names.end(), var_name_str) !=
        output_var_names.end()) {
      var = tvm::relax::Var(vid, NullOpt, NullOpt);
    } else {
      var = tvm::relax::DataflowVar(vid, NullOpt, NullOpt);
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
    return block_builder->EmitMatchShape(tvm::relax::MatchShape(value, pattern, var));
  }
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Emit").set_body_typed(Emit);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.EmitMatchShape").set_body_typed(EmitMatchShape);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
