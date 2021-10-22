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

class VMCompilerImpl : public ExprVisitor {
 public:
  explicit VMCompilerImpl(ExecBuilderNode* builder) {
    builder_ = GetRef<ExecBuilder>(builder);
  }

 protected:
  /*! \brief A counter for naming local functions. */
  int local_func_counter_ = 0;

  // TODO(@yuchen): support visiting other IR nodes
  void VisitExpr_(const FunctionNode* func_node) {
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
    for (auto param : func_node->params) {
      NewRegister(param);
    }
    ExprVisitor::VisitExpr_(func_node);
  }

  void VisitExpr_(const SeqExprNode* op) {
    for (auto block : op->blocks) {
      this->VisitBindingBlock(block);
    }
    // find the function return value and emit the output
    auto ret_reg = this->var_register_map_.find(Downcast<Var>(op->body));
    ICHECK(ret_reg != this->var_register_map_.end());
    builder_->EmitRet(ret_reg->second);
  }

  // TODO: visit call node
  void VisitVarBinding(const VarBinding& binding) {
    Var var = binding->var;
    // TODO(@yuchen): support other nodes than Call
    if (binding->value.as<CallNode>()){
      Call call_node = Downcast<Call>(binding->value);
      if (auto* extern_func = call_node->op.as<relax::ExternFuncNode>()) {
        String name = extern_func->global_symbol;
        if (name == "vm.builtin.alloc_storage") {
          EmitAllocStorage(call_node, var);
        } else if (name == "vm.builtin.alloc_tensor") {
          EmitAllocTensor(call_node, var);
        } else {
          // Normal packed function without attributes
          std::vector<Instruction::Arg> args = ConvertArgs(call_node);
          // TODO(@yuchen): what if the packed func has void return (no need to write to the dst
          // register)?
          builder_->EmitCall(name, args, NewRegister(var));
        }
      } else if (auto* gvar = call_node->op.as<GlobalVarNode>()) {
        String name = gvar->name_hint;
        std::vector<Instruction::Arg> args = ConvertArgs(call_node);
        // TODO: global_var mangling
        builder_->EmitCall(name, args, NewRegister(var));
      } else {
        LOG(FATAL) << "TODO: support compiling everything other than extern functions.";
      }
    } else if (const VarNode* var_node = binding->value.as<VarNode>()) {
      const Var& rhs_var = GetRef<Var>(var_node);
      auto rhs_var_reg = this->var_register_map_.find(rhs_var);
      ICHECK(rhs_var_reg != this->var_register_map_.end());
      this->var_register_map_.insert({var, rhs_var_reg->second});
    } else {
      LOG(FATAL) << "TODO: support compiling everything other than Call and Var.";
    }
  }

  void EmitAllocStorage(const Call& call_node, const Var& var) {
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

    builder_->EmitCall("vm.builtin.alloc_storage", args, NewRegister(var));
  }

  void EmitAllocTensor(const Call& call_node, const Var& var) {
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

    builder_->EmitCall("vm.builtin.alloc_tensor", args, NewRegister(var));
  }

  size_t NewRegister(Var var) {
    size_t reg = this->registers_num_++;
    this->var_register_map_.insert({var, reg});
    return reg;
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

PackedFunc VMCompiler::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "compile") {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 3);
      IRModule mod = args[0];
      this->Compile(mod, args[1], args[2]);
    });
  } else if (name == "get_executable") {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetExec(); });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([name](TVMArgs args, TVMRetValue* rv) {});
  }
}

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

  VMCompilerImpl compiler(builder_.operator->());
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

runtime::Module CreateVMCompiler() {
  auto compiler = make_object<VMCompiler>();
  return runtime::Module(compiler);
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
