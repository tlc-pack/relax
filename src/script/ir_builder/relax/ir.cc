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

////////////////////////////// Tensor Type //////////////////////////////
TensorType::TensorType(Type type, Optional<tvm::relax::Expr> shape) {
  auto n = make_object<TensorTypeNode>();
  n->type = std::move(type);
  n->shape = std::move(shape);
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorTypeNode);

TensorType Tensor(Optional<Array<PrimExpr>> shape, DataType dtype, Optional<Integer> ndim) {
  using namespace tvm::relax;
  int n_dim = -1;
  if (shape.defined() && ndim.defined()) {
    CHECK_EQ(shape.value().size(), ndim.value()->value)
        << "The dimension of the given shape is mismatched with the given `ndim`";
    n_dim = shape.value().size();
  } else if (shape.defined()) {
    n_dim = shape.value().size();
  } else if (ndim.defined()) {
    n_dim = ndim.value()->value;
  } else {
    LOG(FATAL) << "The `ndim` must be specified when the shape is None";
  }
  Type dyn_tensor_type = DynTensorType(n_dim, dtype);
  Optional<Expr> shape_expr = NullOpt;
  if (shape.defined()) {
    shape_expr = ShapeExpr(shape.value());
  }
  return TensorType(dyn_tensor_type, shape_expr);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Tensor").set_body_typed(Tensor);

/////////////////////////////// Function ////////////////////////////////

FunctionFrame Function() {
  ObjectPtr<FunctionFrameNode> n = make_object<FunctionFrameNode>();
  n->block_builder = tvm::relax::BlockBuilder::Create(/*mod=*/NullOpt);
  n->default_binding_block_frame = BindingBlock();
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

tvm::Type RetType(tvm::Type ret_type) {
  FunctionFrame frame = FindFunctionFrame("R.ret_type");
  if (frame->ret_type.defined()) {
    LOG(FATAL) << "ValueError: Duplicate function return type, previous one is:\n "
               << frame->ret_type.value();
  }
  frame->ret_type = ret_type;
  return ret_type;
}

void FuncReturn(const tvm::relax::Expr& value) {
  FunctionFrame frame = FindFunctionFrame("return");
  frame->outputs.push_back(value);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Function").set_body_typed(Function);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.Arg").set_body_typed(Arg);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncName").set_body_typed(FuncName);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncAttrs").set_body_typed(FuncAttrs);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.RetType").set_body_typed(RetType);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.FuncReturn").set_body_typed(FuncReturn);

///////////////////////////// BindingBlock //////////////////////////////

BlockFrame Dataflow() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = true;
  return BlockFrame(n);
}

BlockFrame BindingBlock() {
  ObjectPtr<BlockFrameNode> n = make_object<BlockFrameNode>();
  n->is_dataflow = false;
  return BlockFrame(n);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Dataflow").set_body_typed(Dataflow);
TVM_REGISTER_GLOBAL("script.ir_builder.relax.BindingBlock").set_body_typed(BindingBlock);

/////////////////////////////// Bindings ///////////////////////////////

tvm::relax::Var Emit(const tvm::relax::Expr& expr) {
  tvm::relax::BlockBuilder block_builder = GetBlockBuilder();
  return block_builder->Emit(expr);
}

TVM_REGISTER_GLOBAL("script.ir_builder.relax.Emit").set_body_typed(Emit);

}  // namespace relax
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
