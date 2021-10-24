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
 * \brief A compiler to compile an IRModule to VM executable.
 */

#include "compiler.h"

#include <tvm/target/target.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/function.h>
#include <tvm/driver/driver_api.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace relax_vm {

using namespace relax;

class CodeGenVM : public ExprFunctor<Instruction::Arg(const Expr&)> {
 public:
  explicit CodeGenVM(ExecBuilderNode* builder) {
    builder_ = GetRef<ExecBuilder>(builder);
  }

 protected:
  /*! \brief A counter for naming local functions. */
  int local_func_counter_ = 0;

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
    builder_->EmitRet(ret_reg.data);
    return ret_reg;
  }

  Instruction::Arg VisitExpr_(const CallNode* op) {
    String name;
    if (auto* extern_func = op->op.as<ExternFuncNode>()) {
      name = extern_func->global_symbol;
      if (name == "vm.builtin.alloc_storage") {
        return EmitAllocStorage(GetRef<Call>(op));
      } else if (name == "vm.builtin.alloc_tensor") {
        return EmitAllocTensor(GetRef<Call>(op));
      } 
    } else if (auto* gvar = op->op.as<GlobalVarNode>()) {
      name = gvar->name_hint;
    }
    std::vector<Instruction::Arg> args;
    for (auto arg : op->args) {
      args.push_back(this->VisitExpr(arg));
    }
    builder_->EmitCall(name, args, this->registers_num_);
    return Instruction::Arg(Instruction::kRegister, this->registers_num_++);
  }

  Instruction::Arg VisitExpr_(const VarNode* op) {
    auto it = this->var_register_map_.find(GetRef<Var>(op));
    if (it != this->var_register_map_.end()) {
      return Instruction::Arg(Instruction::kRegister, it->second);
    } else {
      return Instruction::Arg(Instruction::kRegister, this->registers_num_++);
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
    Attrs attrs = call_node->attrs;

    // Get dtype and device_type from the attributes.
    auto alloc_attrs = attrs.as<AllocStorageAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be the AllocStorage attrs";
    DataType dtype = alloc_attrs->dtype;
    int device_type = alloc_attrs->device_type;

    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kVMStateRegister));
    for (Expr arg: call_node->args) {
      args.push_back(ConvertArg(arg));
    }
    args.push_back(Instruction::Arg(Instruction::kImmediate, device_type));

    // store dtype in constant pool
    TVMRetValue data_type;
    data_type = dtype;
    Index index = this->builder_->EmitConstant(data_type);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    builder_->EmitCall("vm.builtin.alloc_storage", args, this->registers_num_);
    return Instruction::Arg(Instruction::kRegister, this->registers_num_++);
  }

  Instruction::Arg EmitAllocTensor(const Call& call_node) {
    Attrs attrs = call_node->attrs;

    // Get dtype from the attributes.
    auto alloc_attrs = attrs.as<AllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be the AllocTensor attrs";
    DataType dtype = alloc_attrs->dtype;

    std::vector<Instruction::Arg> args;
    for (Expr arg: call_node->args) {
      args.push_back(ConvertArg(arg));
    }

    // store dtype in constant pool
    TVMRetValue data_type;
    data_type = dtype;
    Index index = builder_->EmitConstant(data_type);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    builder_->EmitCall("vm.builtin.alloc_tensor", args, this->registers_num_);
    return Instruction::Arg(Instruction::kRegister, this->registers_num_++);
  }

  bool IsConstantShape(ShapeExpr shape) const {
    for (PrimExpr e : shape->values) {
      if (!e->IsInstance<IntImmNode>()) {
        return false;
      }
    }
    return true;
  }

  // TODO: recursive Expr -> instr::arg, ExprFunctor, like llvm builder
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
      LOG(FATAL) << "not supported argument type.";
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

  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
};

void VMCompiler::Compile(IRModule mod, Target target, Target target_host) {
  // Reset internal builder
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
}  // namespace runtime
}  // namespace tvm
