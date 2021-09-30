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
 * \file src/relax/vm/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include "compiler.h"

#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

using namespace relax;

class VMFunctionCompiler : public ExprVisitor {
 public:
  VMFunctionCompiler() { builder_ = ExecBuilderNode::Create(); }

  Executable Get() { return builder_->Get(); }

 protected:
  void VisitExpr_(const FunctionNode* func_node) {
    if (func_node->name.defined()) {
      builder_->Function(func_node->name.value()->name_hint, func_node->params.size());
    } else {
      builder_->Function("local_func", func_node->params.size());
    }
    
    size_t i = 0;
    for (auto param : func_node->params) {
      auto arg_register = NewRegister();
      ICHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      ++i;
    }
    ExprVisitor::VisitExpr_(func_node);
  }

  void VisitExpr_(const SeqExprNode* op) {
    for (auto block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    // find the function return value and emit
    auto ret_reg = this->var_register_map_.find(Downcast<Var>(op->body));
    ICHECK(ret_reg != this->var_register_map_.end());
    builder_->EmitRet(ret_reg->second);
  }

  void VisitVarBinding(const VarBinding& binding) {
    Var var = binding->var;
    Call call_node = Downcast<Call>(binding->value);
    if (auto* extern_func = call_node->op.as<relax::ExternFuncNode>()) {
      String name = extern_func->global_symbol;
      if (name == "vm.builtin.alloc_storage") {
        Attrs attrs = call_node->attrs;

        // Get dtype and device_type from the attributes.
        auto alloc_attrs = attrs.as<AllocStorageAttrs>();
        ICHECK(alloc_attrs != nullptr) << "must be the AllocStorage attrs";
        DataType dtype = alloc_attrs->dtype;
        int device_type = alloc_attrs->device_type;
        PrimExpr size = Downcast<ShapeExpr>(call_node->args[0])->values[0];
        PrimExpr alignment = Downcast<ShapeExpr>(call_node->args[1])->values[0];

        std::vector<Instruction::Arg> args;
        args.push_back(Instruction::Arg(Instruction::kVMStateRegister));
        args.push_back(Instruction::Arg(Instruction::kImmediate, Downcast<IntImm>(size)->value));
        args.push_back(
            Instruction::Arg(Instruction::kImmediate, Downcast<IntImm>(alignment)->value));
        args.push_back(Instruction::Arg(Instruction::kImmediate, device_type));

        // store dtype in constant pool 
        TVMRetValue data_type;
        data_type = dtype;
        Index index = builder_->EmitConstant(data_type);
        args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

        this->var_register_map_.insert({var, this->registers_num_});
        builder_->EmitCall(name, args, NewRegister());
      } else if (name == "vm.builtin.alloc_tensor") {
        Attrs attrs = call_node->attrs;
        auto alloc_attrs = attrs.as<AllocTensorAttrs>();
        ICHECK(alloc_attrs != nullptr) << "must be the AllocTensor attrs";
        DataType dtype = alloc_attrs->dtype;

        std::vector<Instruction::Arg> args;
        auto storage_reg = this->var_register_map_.find(Downcast<Var>(call_node->args[0]));
        ICHECK(storage_reg != this->var_register_map_.end());
        args.push_back(Instruction::Arg(Instruction::kRegister, storage_reg->second));

        PrimExpr offset = Downcast<ShapeExpr>(call_node->args[1])->values[0];
        args.push_back(Instruction::Arg(Instruction::kImmediate, Downcast<IntImm>(offset)->value));

        // store dtype in constant pool 
        TVMRetValue data_type;
        data_type = dtype;
        Index index = builder_->EmitConstant(data_type);
        args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

        // store shape in constant pool 
        std::vector<int64_t> shape;
        auto shape_expr = Downcast<ShapeExpr>(call_node->args[2])->values;
        for (PrimExpr i : shape_expr) {
          shape.push_back(Downcast<IntImm>(i)->value);
        }
        auto shape_tuple = ShapeTuple(shape);
        TVMRetValue shape_tuple_value;
        shape_tuple_value = shape_tuple;
        index = builder_->EmitConstant(shape_tuple_value);
        args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

        this->var_register_map_.insert({var, this->registers_num_});
        builder_->EmitCall(name, args, NewRegister());
      }
      // Normal packed function without attributes
      else {
        std::vector<Instruction::Arg> args_;
        for (size_t i = 0; i < call_node->args.size(); ++i) {
          if (call_node->args[i].as<VarNode>()) {
            auto reg = this->var_register_map_.find(Downcast<Var>(call_node->args[i]));
            ICHECK(reg != this->var_register_map_.end());
            args_.push_back(Instruction::Arg(Instruction::kRegister, reg->second));
          }
        }
        builder_->EmitCall(name, args_, Instruction::kVoidArg);
        // this->var_register_map_.insert({var, this->registers_num_});
        // builder_->EmitCall(name, args_, NewRegister());
        // TODO(yuchen): what if the packed func has void return (no need to write to the dst register)?
      }
    }
  }

  size_t NewRegister() { return registers_num_++; }

  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
};

Executable Compile(const relay::Expr& e) {
  auto compiler = VMFunctionCompiler();
  compiler.VisitExpr(e);
  return compiler.Get();
}

TVM_REGISTER_GLOBAL("relax.transform.compile").set_body_typed([](relay::Expr expr) {
  return Compile(expr);
});

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
