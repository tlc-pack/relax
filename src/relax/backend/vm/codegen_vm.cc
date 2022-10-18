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
 * \brief A codegen to generate VM executable from a Relax IRModule.
 */

#include "codegen_vm.h"

#include <tvm/driver/driver_api.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../../../target/metadata_module.h"
#include "../../../target/source/codegen_source_base.h"

namespace tvm {
namespace relax {
namespace relax_vm {

using namespace relax;

// Helper function to get the function name of the registered packed function implementation of
// relax operator.
FCallPacked GetPackedFuncName(const Call& call) {
  auto op_map = Op::GetAttrMap<FCallPacked>("FCallPacked");
  if (call->op.as<OpNode>()) {
    Op op = Downcast<Op>(call->op);
    if (op_map.count(op)) {
      return op_map[op];
    }
  }
  return {};
}

/*!
 * \brief A class to generate VM executable for Relax functions.
 */
class CodeGenVM : public ExprFunctor<Instruction::Arg(const Expr&)> {
 public:
  explicit CodeGenVM(ExecBuilderNode* builder) { builder_ = GetRef<ExecBuilder>(builder); }

 protected:
  size_t NewRegister() { return registers_num_++; }
  Instruction::Arg VisitExpr_(const ExternFuncNode* func) {
    const static constexpr char* kCSource = "c_source";
    const static constexpr char* kCSourceFmt = "c_source_fmt";
    if (Optional<String> opt_code = func->attrs.GetAttr<String>(kCSource)) {
      String sym = func->global_symbol;
      String fmt = func->attrs.GetAttr<String>(kCSourceFmt).value_or("c");
      String code = opt_code.value();
      Module c_source_module =
          codegen::CSourceModuleCreate(/*code=*/code, /*fmt=*/fmt, /*func_names=*/{sym},
                                       /*const_vars=*/{});
      builder_->exec->Import(c_source_module);
    }
    return Instruction::Arg{};
  }

  Instruction::Arg VisitExpr_(const FunctionNode* func_node) {
    Optional<String> gsymbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "there should be no local functions in Relax VM codegen phase. "
                                 "Did you forget to apply LambdaLift pass?";

    Array<String> param_names;
    for (Var param : func_node->params) {
      param_names.push_back(param->name_hint());
    }

    builder_->EmitFunction(gsymbol.value(), func_node->params.size(), param_names);

    for (Var param : func_node->params) {
      Instruction::Arg reg = this->VisitExpr(param);
      this->var_register_map_.insert({param, reg.data});
    }
    Instruction::Arg ret = ExprFunctor::VisitExpr(func_node->body);
    builder_->EmitRet(ret.data);
    registers_num_ = 0;
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

  Instruction::Arg VisitExpr_(const CallNode* call_node) {
    if (call_node->op.as<OpNode>()) {
      // special case generate for the intrinsics whose attribute fields
      // cannot be represented by args in the CallNode
      const Call& call = GetRef<Call>(call_node);
      FCallPacked name = GetPackedFuncName(call);
      if (!name.empty()) {
        // If the operator has a registered packed function implementation, emit call to that packed
        // function.
        return EmitPackedFuncCall(call, name);
      } else if (call_node->op == alloc_storage_op_) {
        return EmitAllocStorage(call);
      } else if (call_node->op == alloc_tensor_op_) {
        return EmitAllocTensor(call);
      } else if (call_node->op == store_shape_op_ || call_node->op == load_shape_op_) {
        return EmitShape(call);
      } else if (call_node->op == call_tir_dyn_op_) {
        return EmitTirDynOp(call);
      } else if (call_node->op == make_closure_op_) {
        return EmitAllocClosure(call);
      } else if (call_node->op == invoke_closure_op_) {
        return EmitInvokeClosure(call);
      } else {
        // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
        LOG(FATAL) << "CodeGenVM cannot handle this intrinsic now:\n" << call_node->op;
      }
    }
    String name;
    if (auto* extern_func = call_node->op.as<ExternFuncNode>()) {
      name = extern_func->global_symbol;
    } else if (auto* gvar = call_node->op.as<GlobalVarNode>()) {
      // GlobalVar can be reference to a Relax function or a TIR primfunc
      name = gvar->name_hint;
    } else {
      LOG(FATAL) << "CodeGenVM does not support calls to " << call_node->op->GetTypeKey();
    }
    std::vector<Instruction::Arg> args;
    // For extern function `vm.builtin.alloc_shape_heap` we must pass vm register as the first
    // argument to find the device in which shape heap should be allocated.
    if (name == "vm.builtin.alloc_shape_heap") {
      args.push_back(Instruction::Arg(Instruction::kRegister, Instruction::kVMRegister));
    }
    std::vector<Instruction::Arg> converted_args = ConvertArgs(GetRef<Call>(call_node));
    args.insert(args.end(), converted_args.begin(), converted_args.end());
    size_t dst_register = NewRegister();
    builder_->EmitCall(name, args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
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

    size_t dst_register = NewRegister();
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));
    builder_->EmitCall("vm.builtin.copy", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
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
    size_t dst_register = NewRegister();
    builder_->EmitCall("runtime.Tuple", args, dst_register);

    return Instruction::Arg(Instruction::kRegister, dst_register);
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

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.runtime.TupleGetItem", args, dst_register);

    return Instruction::Arg(Instruction::kRegister, dst_register);
  }

  Instruction::Arg EmitAllocStorage(const Call& call_node) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kVMRegister));
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

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.alloc_storage", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
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
    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.alloc_tensor", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
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
      indices_vec.push_back(ind.IntValue());
    }
    ShapeTuple indices = ShapeTuple(indices_vec);
    TVMRetValue indices_const;
    indices_const = indices;
    Index index = builder_->EmitConstant(indices_const);
    args.push_back(Instruction::Arg(Instruction::kConstIdx, index));

    size_t dst_register = NewRegister();
    if (call_node->op == store_shape_op_) {
      builder_->EmitCall("vm.builtin.store_shape", args, dst_register);
    } else if (call_node->op == load_shape_op_) {
      builder_->EmitCall("vm.builtin.load_shape", args, dst_register);
    }
    return Instruction::Arg(Instruction::kRegister, dst_register);
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
    args.push_back(Instruction::Arg(Instruction::kVMRegister));
    args.push_back(Instruction::Arg(Instruction::kConstIdx, func_name_index));
    for (Expr arg : tir_args->fields) {
      args.push_back(ConvertArg(arg));
    }

    size_t dst_register = NewRegister();

    builder_->EmitCall("vm.call_tir_dyn", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
  }

  template <typename T>
  Instruction::Arg EmitConstantFromValue(T value) {
    TVMRetValue tvm_value;
    tvm_value = value;
    Index index = builder_->EmitConstant(tvm_value);
    return Instruction::Arg(Instruction::kConstIdx, index);
  }

  // Emit the `call_node` attributes as constants and append these constants to `args` vector.
  void AppendAttrsAsConstants(const Call& call_node, std::vector<Instruction::Arg>& args) {
    auto attrs = call_node->attrs;
    if (!attrs.defined()) return;

    if (call_node->op == unique_op_) {
      auto unique_attrs = call_node->attrs.as<UniqueAttrs>();
      args.push_back(EmitConstantFromValue(unique_attrs->sorted));
      args.push_back(EmitConstantFromValue(unique_attrs->return_inverse));
      args.push_back(EmitConstantFromValue(unique_attrs->return_counts));
      args.push_back(EmitConstantFromValue(unique_attrs->dim));
      return;
    }
    if (call_node->op == print_op_) {
      auto print_attrs = call_node->attrs.as<PrintAttrs>();
      // format string is the first argument
      args.insert(args.begin(), EmitConstantFromValue(print_attrs->format));
      return;
    }
    if (call_node->op == assert_op_) {
      auto assert_attrs = call_node->attrs.as<AssertOpAttrs>();
      // format string comes before the format args
      args.insert(args.begin() + 1, EmitConstantFromValue(assert_attrs->format));
      return;
    }
    LOG(FATAL) << "Support for attributes of Op " << call_node->op
               << " has not been implemented yet.";
    return;
  }

  // Emits call to packed function `name` with arguments copied over from `call_node` args and
  // attributes.
  Instruction::Arg EmitPackedFuncCall(const Call& call_node, const FCallPacked& name) {
    std::vector<Instruction::Arg> args;
    args = ConvertArgs(call_node);
    AppendAttrsAsConstants(call_node, args);
    size_t dst_register = NewRegister();
    builder_->EmitCall(name, args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
  }

  Instruction::Arg EmitAllocClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    auto gv = Downcast<GlobalVar>(call_node->args[0]);
    auto closure_args = Downcast<Tuple>(call_node->args[1]);
    auto func_name = gv->name_hint;

    TVMRetValue func_name_constant;
    func_name_constant = func_name;
    auto func_name_index = builder_->EmitConstant(func_name_constant);

    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg(Instruction::kConstIdx, func_name_index));
    for (Expr arg : closure_args->fields) {
      args.push_back(ConvertArg(arg));
    }

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.alloc_closure", args, dst_register);
    return Instruction::Arg(Instruction::kRegister, dst_register);
  }

  Instruction::Arg EmitInvokeClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<VarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    std::vector<Instruction::Arg> args;
    // VM is utilized to help get the Function in builtin packedfunc
    args.push_back(Instruction::Arg(Instruction::kVMRegister));

    auto lv = Downcast<Var>(call_node->args[0]);
    auto it = this->var_register_map_.find(lv);
    if (it != this->var_register_map_.end()) {
      args.push_back(Instruction::Arg(Instruction::kRegister, it->second));
    } else {
      args.push_back(Instruction::Arg(Instruction::kRegister, registers_num_));
    }

    // args for the invoke_closure
    auto invoke_closure_args = Downcast<Tuple>(call_node->args[1]);
    for (Expr arg : invoke_closure_args->fields) {
      args.push_back(ConvertArg(arg));
    }

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.invoke_closure", args, dst_register);
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
    } else if (arg->IsInstance<ConstantNode>()) {
      TVMRetValue constant_data;
      constant_data = Downcast<Constant>(arg)->data;
      Index index = builder_->EmitConstant(constant_data);
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
  const Op& unique_op_ = Op::Get("relax.unique");
  const Op& print_op_ = Op::Get("relax.print");
  const Op& assert_op_ = Op::Get("relax.assert_op");
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
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
Module CodeGen(IRModule mod, Optional<Module> lib, Array<Module> ext_libs, Target target,
               Map<String, runtime::NDArray> params) {
  VMCodeGen codegen;
  codegen.CodeGen(mod);
  ObjectPtr<Executable> executable = codegen.GetExec();
  if (!lib.defined()) {
    lib = codegen::CSourceModuleCreate(";", "", Array<String>{});
  }
  std::unordered_map<std::string, runtime::NDArray> conv_params;
  for (const auto& kv : params) {
    conv_params[kv.first] = kv.second;
  }
  Module combined_lib = codegen::CreateMetadataModule(
      conv_params, lib.value(), ext_libs, target,

      // TODO(@sunggg): Currently, CRT uses relay-specific executor for uTVM support.
      // Before jumping into details, only support cpp runtime for now.
      relay::Runtime::Create("cpp"),
      relay::Executor::Create("graph"),  // TODO(@sunggg): pass arbitrarily executor. CPP runtime
                                         // won't use this anyways.
      relay::backend::ExecutorCodegenMetadata());
  executable->Import(combined_lib);
  return Module(executable);
}

TVM_REGISTER_GLOBAL("relax.VMCodeGen").set_body_typed(CodeGen);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
