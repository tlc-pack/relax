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
 * \file src/relax/backend/vm/codegen_vm.cc
 * \brief A compiler to compile an IRModule to VM executable.
 */

#include "codegen_vm.h"

#include <tvm/target/target.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/driver/driver_api.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace relax_vm {

using namespace relax;

/*!
 * \brief A class to generate VM executable for Relax functions.
 */
class CodeGenVM : public ExprFunctor<Instruction::Arg(const Expr&)> {
 public:
  explicit CodeGenVM(ExecBuilderNode* builder) {
    builder_ = GetRef<ExecBuilder>(builder);
  }

 protected:
  size_t NewRegister() { return registers_num_++; }

  // TODO(@yuchen): add visitors for IfNode when goto and if instructions are introduced to relax vm.

  // TODO(@yuchen): when we support closure, this visitor should return a register that 
  // contains the closure object.
  Instruction::Arg VisitExpr_(const FunctionNode* func_node) {
    if (func_node->name.defined()) {
      builder_->EmitFunction(func_node->name.value()->name_hint, func_node->params.size());
    } else {
      // TODO(@yuchen): handle local functions that capture local vars outside the func
      // TODO(@yuchen): a renaming pass to resolve name conflicts, e.g. the input module has a
      // function named "local_funcN"
      // lift the local func to a global func and compile it normally
      builder_->EmitFunction("local_func" + std::to_string(local_func_counter_++),
                             func_node->params.size());
    }
    for (Var param : func_node->params) {
      Instruction::Arg reg = this->VisitExpr(param);
      this->var_register_map_.insert({param, reg.data});
    }
    Instruction::Arg ret = ExprFunctor::VisitExpr(func_node->body);
    builder_->EmitRet(ret.data);
    return ret;
  }

  Instruction::Arg VisitExpr_(const SeqExprNode* op) {
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        ICHECK(binding->IsInstance<VarBindingNode>());
        Expr value = Downcast<VarBinding>(binding)->value;
        Var var = Downcast<VarBinding>(binding)->var;
        Instruction::Arg reg = this->VisitExpr(value);
        this->var_register_map_.insert({var, reg.data});
      }
    }

    Instruction::Arg ret_reg = this->VisitExpr(op->body);
    return ret_reg;
  }

  Instruction::Arg VisitExpr_(const CallNode* op) {
    if (op->op.as<OpNode>()) {
      // special case generate for the intrinsics whose attribute fields 
      // cannot be represented by args in the CallNode
      const Call& call = GetRef<Call>(op);
      if (op->op == alloc_storage_op_) {
        return EmitAllocStorage(call);
      } else if (op->op == alloc_tensor_op_) {
        return EmitAllocTensor(call);
      } else if (op->op == store_shape_op_ || op->op == load_shape_op_) {
        return EmitShape(call);
      } else {
        // every "normal" operator is lowered to a global var in the IR module. The Attrs for those ops 
        // are handled in a pass when lowering them to TIR.
        LOG(FATAL) << "CodeGenVM cannot handle this intrinsic now:\n" << op->op;
      }
    }
    String name;
    if (auto* extern_func = op->op.as<ExternFuncNode>()) {
      name = extern_func->global_symbol;
    } else if (auto* gvar = op->op.as<GlobalVarNode>()) {
      name = gvar->name_hint;
    } else {
      LOG(FATAL) << "CodeGenVM does not support calls to " << op->op->GetTypeKey();
    }
    std::vector<Instruction::Arg> args;
    for (auto arg : op->args) {
      args.push_back(this->VisitExpr(arg));
    }
    size_t arg_register = NewRegister();
    builder_->EmitCall(name, args, arg_register);

    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  Instruction::Arg VisitExpr_(const VarNode* op) {
    auto it = this->var_register_map_.find(GetRef<Var>(op));
    if (it != this->var_register_map_.end()) {
      return Instruction::Arg(Instruction::kRegister, it->second);
    } else {
      return Instruction::Arg(Instruction::kRegister, NewRegister());
    }
  }

  Instruction::Arg VisitExpr_(const ShapeExprNode* op) {
    ShapeExpr sh = GetRef<ShapeExpr>(op);
    ICHECK(IsConstantShape(sh))
      << "should only use constant shape after shape lowering: "
      << sh->values;
    std::vector<int64_t> shape;
    for (PrimExpr e : sh->values) {
      shape.push_back(Downcast<IntImm>(e)->value);
    }
    auto shape_tuple = ShapeTuple(shape);
    TVMRetValue shape_tuple_value;
    shape_tuple_value = shape_tuple;
    Index index = builder_->EmitConstant(shape_tuple_value);
    return Instruction::Arg(Instruction::kConstIdx, index);
  }

  Instruction::Arg EmitAllocStorage(const Call& call_node) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kVMStateRegister));
    for (Expr arg: call_node->args) {
      args.push_back(ConvertArg(arg));
    }

    // Handle attrs of the call
    auto alloc_attrs = call_node->attrs.as<AllocStorageAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be AllocStorageAttrs";
    int device_type = alloc_attrs->device_type;
    args.push_back(Instruction::Arg(Instruction::kImmediate, device_type));
    DataType dtype = alloc_attrs->dtype;
    TVMRetValue data_type;
    data_type = dtype;
    Index index = this->builder_->EmitConstant(data_type);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    size_t arg_register = NewRegister();
    builder_->EmitCall("vm.builtin.alloc_storage", args, arg_register);
    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  Instruction::Arg EmitAllocTensor(const Call& call_node) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    for (Expr arg: call_node->args) {
      args.push_back(ConvertArg(arg));
    }

    // Handle attrs of the call
    auto alloc_attrs = call_node->attrs.as<AllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be AllocTensorAttrs";
    int offset = alloc_attrs->offset;
    args.push_back(Instruction::Arg(Instruction::kImmediate, offset));
    DataType dtype = alloc_attrs->dtype;
    TVMRetValue data_type;
    data_type = dtype;
    Index index = this->builder_->EmitConstant(data_type);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    size_t arg_register = NewRegister();
    builder_->EmitCall("vm.builtin.alloc_tensor", args, arg_register);
    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  Instruction::Arg EmitShape(const Call& call_node) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    for (Expr arg: call_node->args) {
      args.push_back(ConvertArg(arg));
    }

    // Handle attrs of the call
    auto shape_attrs = call_node->attrs.as<ShapeHeapAttrs>();
    ICHECK(shape_attrs != nullptr) << "must be ShapeHeapAttrs";
    std::vector<int64_t> indices_vec;
    for (Integer ind : shape_attrs->indices) {
      indices_vec.push_back(ind);
    }
    ShapeTuple indices = ShapeTuple(indices_vec);
    TVMRetValue indices_const;
    indices_const = indices;
    Index index = builder_->EmitConstant(indices_const);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    size_t arg_register = NewRegister();
    if (call_node->op == store_shape_op_) {
      builder_->EmitCall("vm.builtin.store_shape", args, arg_register);
    } else if (call_node->op == load_shape_op_) {
      builder_->EmitCall("vm.builtin.load_shape", args, arg_register);
    }
    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  bool IsConstantShape(ShapeExpr shape) const {
    for (PrimExpr e : shape->values) {
      if (!e->IsInstance<IntImmNode>()) {
        return false;
      }
    }
    return true;
  }

  Instruction::Arg ConvertArg(Expr arg) {
    if (arg->IsInstance<VarNode>()) {
      Var var = Downcast<Var>(arg);
      auto reg = this->var_register_map_.find(Downcast<Var>(arg));
      ICHECK(reg != this->var_register_map_.end())
        << var->name_hint() << "(" << var << ")" << " not in the register map.";
      return Instruction::Arg(Instruction::kRegister, reg->second);
    } else if (arg->IsInstance<ShapeExprNode>()) {
      ShapeExpr sh = Downcast<ShapeExpr>(arg);
      ICHECK(IsConstantShape(sh))
        << "should only use constant shape after shape lowering: "
        << sh->values;
      std::vector<int64_t> shape;
      for (PrimExpr e : sh->values) {
        shape.push_back(Downcast<IntImm>(e)->value);
      }
      auto shape_tuple = ShapeTuple(shape);
      TVMRetValue shape_tuple_value;
      shape_tuple_value = shape_tuple;
      Index index = builder_->EmitConstant(shape_tuple_value);
      return Instruction::Arg(Instruction::kConstIdx, index);
    } else {
      LOG(FATAL) << "CodeGenVM does not this argument type:\n" << arg->GetTypeKey();
    }
    return Instruction::Arg();
  }

  std::vector<Instruction::Arg> ConvertArgs(const Call& call) {
    std::vector<Instruction::Arg> ret;
    for (size_t i = 0; i < call->args.size(); ++i) {
      ret.push_back(ConvertArg(call->args[i]));
    }
    return ret;
  }

  /*! \brief A counter for naming local functions. */
  int local_func_counter_ = 0;
  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  const Op& alloc_storage_op_ = Op::Get("relax.vm.builtin.alloc_storage");
  const Op& alloc_tensor_op_ = Op::Get("relax.vm.builtin.alloc_tensor");
  const Op& store_shape_op_ = Op::Get("relax.vm.builtin.store_shape");
  const Op& load_shape_op_ = Op::Get("relax.vm.builtin.load_shape");
};

void VMCompiler::Compile(IRModule mod, Target target, Target target_host) {
  builder_ = relax::ExecBuilderNode::Create();

  IRModule tir_mod;
  IRModule rx_mod;
  for (auto& p : mod->functions) {
    auto gvar = p.first;

    BaseFunc func = p.second;
    if (func.as<tir::PrimFuncNode>()) {
      tir_mod->Add(gvar, func);
    } else if (func.as<FunctionNode>()) {
      rx_mod->Add(gvar, func);
    } else {
      LOG(FATAL) << "Cannot handle such function node now:\n" << func;
    }
  }
  lib_ = tvm::build(tir_mod, target, target_host);

  CodeGenVM compiler(builder_.operator->());
  for (auto& p : rx_mod->functions) {
    compiler.VisitExpr(p.second);
  }
}

Executable VMCompiler::GetExec() { 
  return builder_->Get();
}

runtime::Module VMCompiler::GetLib() {
  return lib_;
}

Array<ObjectRef> Build(IRModule mod, Target target, Target target_host) {
  auto compiler = make_object<VMCompiler>();
  compiler->Compile(mod, target, target_host);
  Executable exec = compiler->GetExec();
  Module lib = compiler->GetLib();
  return Array<ObjectRef>({exec, lib});
}

TVM_REGISTER_GLOBAL("relax.VMBuild")
.set_body_typed(Build);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
