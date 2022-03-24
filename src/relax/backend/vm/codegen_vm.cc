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
 * \brief A codegen to generate VM executable from an IRModule with relax functions.
 */

#include "codegen_vm.h"

#include <tvm/driver/driver_api.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

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
  explicit CodeGenVM(ExecBuilderNode* builder) { builder_ = GetRef<ExecBuilder>(builder); }

 protected:
  size_t NewRegister() { return registers_num_++; }
  // TODO(@yuchen): when we support closure, this visitor should return a register that
  // contains the closure object.
  Instruction::Arg VisitExpr_(const FunctionNode* func_node) {
    if (func_node->name.defined()) {
      builder_->EmitFunction(func_node->name.value()->name_hint, func_node->params.size());
    } else {
      // TODO(@yuchen): handle local functions that capture local vars outside the func
      // TODO(@yuchen): a renaming pass to resolve name conflicts, e.g. the input module has a
      // function named "local_funcN"
      // lift the local func to a global func and process it normally
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
      } else if (op->op == call_tir_dyn_op_) {
        return EmitTirDynOp(call);
      } else {
        // every "normal" operator is lowered to a global var in the IR module. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
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

  Instruction::Arg VisitExpr_(const IfNode* op) {
    const If& ife = GetRef<If>(op);
    // Get the executable from exec_builder
    ObjectPtr<Executable> exec_ = builder_->Get();

    // Visit the condition expression
    Instruction::Arg cond_reg = this->VisitExpr(ife->cond);
    // Record the offset of If instruction
    size_t if_offset = exec_->instr_offset.size();

    builder_->EmitIf(cond_reg.value(), 3);
    size_t num_instr = exec_->instr_offset.size();
    Instruction::Arg true_reg = this->VisitExpr(ife->true_branch);
    // Reserve a register for return
    size_t merge_register = NewRegister();
    // Copy the output from true branch to merge register
    builder_->EmitCall("vm.builtin.copy", {true_reg}, merge_register);

    // Record the offset of Goto instruction
    size_t goto_offset = exec_->instr_offset.size();

    builder_->EmitGoto(1);

    // Calculate the false offset of If
    size_t false_offset = exec_->instr_offset.size() - num_instr + 1;

    Instruction::Arg false_reg = this->VisitExpr(ife->false_branch);
    // Copy the output data of false branch to merge register
    builder_->EmitCall("vm.builtin.copy", {false_reg}, merge_register);

    // Update the offsets of the If instruction emitted above
    // Jump to the behind of the next goto instruction
    exec_->SetInstructionData(if_offset, 2, static_cast<ExecWord>(false_offset));
    // Update the pc_offset of Goto instruction
    // Jump over the false branch
    size_t pc_offset = exec_->instr_offset.size() - goto_offset;
    exec_->SetInstructionData(goto_offset, 1, static_cast<ExecWord>(pc_offset));
    return Instruction::Arg(Instruction::kRegister, merge_register);
  }

  Instruction::Arg VisitExpr_(const VarNode* op) {
    auto it = this->var_register_map_.find(GetRef<Var>(op));
    if (it != this->var_register_map_.end()) {
      return Instruction::Arg(Instruction::kRegister, it->second);
    } else {
      return Instruction::Arg(Instruction::kRegister, NewRegister());
    }
  }

  Instruction::Arg VisitExpr_(const ConstantNode* op) {
    TVMRetValue constant_data;
    constant_data = op->data;
    Index index = this->builder_->EmitConstant(constant_data);
    return Instruction::Arg(Instruction::kConstIdx, index);
  }

  Instruction::Arg VisitExpr_(const ShapeExprNode* op) {
    ShapeExpr sh = GetRef<ShapeExpr>(op);
    ICHECK(IsConstantShape(sh)) << "should only use constant shape after shape lowering: "
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

  Instruction::Arg VisitExpr_(const TupleNode* op) {
    Tuple tuple = GetRef<Tuple>(op);
    std::vector<Instruction::Arg> args;
    for (auto arg : tuple->fields) {
      args.push_back(this->VisitExpr(arg));
    }
    size_t arg_register = NewRegister();
    builder_->EmitCall("runtime.Tuple", args, arg_register);

    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  Instruction::Arg VisitExpr_(const TupleGetItemNode* op) {
    TupleGetItem expr = GetRef<TupleGetItem>(op);
    std::vector<Instruction::Arg> args = {this->VisitExpr(expr->tuple)};

    std::vector<int64_t> tuple_index = {expr->index};
    auto shape_tuple = ShapeTuple(tuple_index);
    TVMRetValue shape_tuple_value;
    shape_tuple_value = shape_tuple;
    Index index = builder_->EmitConstant(shape_tuple_value);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    size_t arg_register = NewRegister();
    builder_->EmitCall("vm.runtime.TupleGetItem", args, arg_register);

    return Instruction::Arg(Instruction::kRegister, arg_register);
  }

  Instruction::Arg EmitAllocStorage(const Call& call_node) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kVMStateRegister));
    for (Expr arg : call_node->args) {
      args.push_back(ConvertArg(arg));
    }

    // Handle attrs of the call
    auto alloc_attrs = call_node->attrs.as<VMAllocStorageAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocStorageAttrs";
    Index runtime_device_index = alloc_attrs->runtime_device_index;
    args.push_back(Instruction::Arg(Instruction::kImmediate, runtime_device_index));
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
    ICHECK_EQ(call_node->args.size(), 2);
    std::vector<Instruction::Arg> args;
    args.reserve(4);
    // Handle `self`
    args.push_back(ConvertArg(call_node->args[0]));
    // Handle `offset`
    auto alloc_attrs = call_node->attrs.as<VMAllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocTensorAttrs";
    int offset = alloc_attrs->offset;
    args.push_back(Instruction::Arg(Instruction::kImmediate, offset));
    // Handle `shape`
    args.push_back(ConvertArg(call_node->args[1]));
    // Handle `dtype`
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
    for (Expr arg : call_node->args) {
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

  Instruction::Arg EmitTirDynOp(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    auto gv = Downcast<GlobalVar>(call_node->args[0]);
    auto tir_args = Downcast<Tuple>(call_node->args[1]);
    auto func_name = gv->name_hint;

    TVMRetValue func_name_constant;
    func_name_constant = func_name;
    auto func_name_index = builder_->EmitConstant(func_name_constant);

    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kVMStateRegister));
    args.push_back(Instruction::Arg(Instruction::kConstIdx, func_name_index));
    for (Expr arg : tir_args->fields) {
      args.push_back(ConvertArg(arg));
    }

    size_t dst_register = NewRegister();

    builder_->EmitCall("vm.call_tir_dyn", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
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
      ICHECK(reg != this->var_register_map_.end()) << var->name_hint() << "(" << var << ")"
                                                   << " not in the register map.";
      return Instruction::Arg(Instruction::kRegister, reg->second);
    } else if (arg->IsInstance<ShapeExprNode>()) {
      ShapeExpr sh = Downcast<ShapeExpr>(arg);
      ICHECK(IsConstantShape(sh)) << "should only use constant shape after shape lowering: "
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
      LOG(FATAL) << "CodeGenVM does not support this argument type:\n" << arg->GetTypeKey();
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
  size_t local_func_counter_ = 0;
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
  const Op& call_tir_dyn_op_ = Op::Get("relax.vm.call_tir_dyn");
};

void VMCodeGen::CodeGen(IRModule rx_mod) {
  builder_ = relax::ExecBuilderNode::Create();
  CodeGenVM codegen(builder_.operator->());
  for (auto& p : rx_mod->functions) {
    codegen.VisitExpr(p.second);
  }
}

ObjectPtr<Executable> VMCodeGen::GetExec() { return builder_->Get(); }

/*!
 * \brief Create the Relax VM executable from an IRModule of Relax function(s) and, possibly, a
 * kernel library.
 * \param mod The IRModule containing Relax function(s).
 * \param lib The kernel library.
 * \return The constructed Relax VM executable.
 */
Module CodeGen(IRModule mod, Optional<Module> lib) {
  VMCodeGen codegen;
  codegen.CodeGen(mod);
  ObjectPtr<Executable> executable = codegen.GetExec();
  if (lib.defined()) {
    executable->Import(lib.value());
  }
  return Module(executable);
}

TVM_REGISTER_GLOBAL("relax.VMCodeGen").set_body_typed(CodeGen);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
