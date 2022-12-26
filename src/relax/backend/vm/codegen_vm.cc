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
#include <tvm/relax/attrs/builtin.h>
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
  explicit CodeGenVM(relax::ExecBuilder builder, IRModule ctx_mod)
      : builder_(builder), ctx_mod_(ctx_mod) {}

 protected:
  size_t NewRegister() { return registers_num_++; }

  // Convert Arg value to a register, trigger copy if needed
  Instruction::Arg EnsureReg(Instruction::Arg arg) {
    if (arg.kind() == Instruction::ArgKind::kRegister) {
      return arg;
    } else {
      RegName dst_reg = NewRegister();
      builder_->EmitCall("vm.builtin.copy", {arg}, dst_reg);
      return Instruction::Arg::Register(dst_reg);
    }
  }

  Instruction::Arg VisitExpr_(const FunctionNode* func_node) {
    Optional<String> gsymbol = func_node->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "there should be no local functions in Relax VM codegen phase. "
                                 "Did you forget to apply LambdaLift or AttachGlobalSymbol Pass?";

    Array<String> param_names;
    for (Var param : func_node->params) {
      param_names.push_back(param->name_hint());
    }

    builder_->EmitFunction(gsymbol.value(), func_node->params.size(), param_names);

    for (size_t i = 0; i < func_node->params.size(); ++i) {
      RegName r = NewRegister();
      ICHECK_EQ(r, static_cast<RegName>(i));
      this->var_arg_map_.insert({func_node->params[i], Instruction::Arg::Register(r)});
    }
    Instruction::Arg ret = ExprFunctor::VisitExpr(func_node->body);
    builder_->EmitRet(EnsureReg(ret));
    registers_num_ = 0;
    return ret;
  }

  Instruction::Arg VisitExpr_(const SeqExprNode* op) {
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        Instruction::Arg value;
        if (auto* var_binding = binding.as<VarBindingNode>()) {
          value = this->VisitExpr(var_binding->value);
        } else if (auto* match_cast = binding.as<MatchCastNode>()) {
          value = this->VisitExpr(match_cast->value);
        } else {
          LOG(FATAL) << "Unsupported binding " << binding->GetTypeKey();
        }
        this->var_arg_map_.insert({binding->var, value});
      }
    }

    Instruction::Arg ret_reg = this->VisitExpr(op->body);
    return ret_reg;
  }

  Instruction::Arg VisitExpr_(const CallNode* call_node) {
    Call call = GetRef<Call>(call_node);

    if (call_node->op == null_value_op_) {
      return Instruction::Arg::Register(Instruction::kVoidRegister);
    }

    // allocate dst register.
    RegName dst_reg = HasVoidStructInfo(call) ? Instruction::kVoidRegister : NewRegister();
    if (call->op.as<OpNode>()) {
      // special case generate for the intrinsics whose attribute fields
      // cannot be represented by args in the CallNode
      FCallPacked name = GetPackedFuncName(call);
      if (!name.empty()) {
        // If the operator has a registered packed function implementation, emit call to that packed
        // function.
        EmitPackedFuncCall(call, name, dst_reg);
      } else if (call_node->op == call_builtin_op_) {
        // TODO(relax-team) migrate most handling of op to
        // directly map to call_builtin before codegen and simplify vm codegen.
        EmitCallBuiltin(call, dst_reg);
      } else if (call_node->op == alloc_storage_op_) {
        EmitAllocStorage(call, dst_reg);
      } else if (call_node->op == alloc_tensor_op_) {
        EmitAllocTensor(call, dst_reg);
      } else if (call_node->op == call_tir_dyn_op_) {
        EmitTirDynOp(call, dst_reg);
      } else if (call_node->op == make_closure_op_) {
        EmitAllocClosure(call, dst_reg);
      } else if (call_node->op == invoke_closure_op_) {
        EmitInvokeClosure(call, dst_reg);
      } else {
        // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
        LOG(FATAL) << "CodeGenVM cannot handle this intrinsic now:\n" << call_node->op;
      }
    } else {
      EmitNormalCall(call, dst_reg);
    }
    return Instruction::Arg::Register(dst_reg);
  }

  Instruction::Arg VisitExpr_(const IfNode* op) {
    const If& ife = GetRef<If>(op);

    // Visit the condition expression
    // NOTE: must call ensure reg here so we won't have extra flags
    Instruction::Arg cond_reg = EnsureReg(this->VisitExpr(ife->cond));

    // obtain the temp exec in progress.
    vm::Executable* exec = builder_->exec();

    // Record the offset of If instruction
    size_t if_offset = exec->instr_offset.size();

    builder_->EmitIf(cond_reg, 3);
    size_t num_instr = exec->instr_offset.size();
    Instruction::Arg true_value = this->VisitExpr(ife->true_branch);
    // Reserve a register for return
    size_t merge_register = NewRegister();
    // Copy the output from true branch to merge register
    builder_->EmitCall("vm.builtin.copy", {true_value}, merge_register);

    // Record the offset of Goto instruction
    size_t goto_offset = exec->instr_offset.size();

    builder_->EmitGoto(1);

    // Calculate the false offset of If
    size_t false_offset = exec->instr_offset.size() - num_instr + 1;

    Instruction::Arg false_falue = this->VisitExpr(ife->false_branch);
    // Copy the output data of false branch to merge register
    builder_->EmitCall("vm.builtin.copy", {false_falue}, merge_register);

    // Update the offsets of the If instruction emitted above
    // Jump to the behind of the next goto instruction
    exec->SetInstructionData(if_offset, 2, static_cast<ExecWord>(false_offset));
    // Update the pc_offset of Goto instruction
    // Jump over the false branch
    size_t pc_offset = exec->instr_offset.size() - goto_offset;
    exec->SetInstructionData(goto_offset, 1, static_cast<ExecWord>(pc_offset));
    return Instruction::Arg::Register(merge_register);
  }

  Instruction::Arg VisitExpr_(const VarNode* op) {
    Var var = GetRef<Var>(op);
    auto it = this->var_arg_map_.find(var);
    ICHECK(it != this->var_arg_map_.end()) << "Var " << var << " is not defined";
    return it->second;
  }

  Instruction::Arg VisitExpr_(const ExternFuncNode* op) {
    // TODO(relax-team) turn into get function builtin.
    LOG(FATAL) << "ExternFunc cannot appear directly in args, use call_builtin instead";
    return Instruction::Arg::Register(Instruction::kVoidRegister);
  }

  Instruction::Arg VisitExpr_(const ConstantNode* op) {
    return builder_->ConvertConstant(op->data);
  }

  Instruction::Arg VisitExpr_(const ShapeExprNode* op) {
    std::vector<int64_t> shape;
    for (PrimExpr e : op->values) {
      if (auto* int_value = e.as<IntImmNode>()) {
        shape.push_back(int_value->value);
      } else {
        LOG(FATAL) << "Should only use constant shape after shape lowering: " << op->values;
      }
    }
    return builder_->ConvertConstant(ShapeTuple(shape));
  }

  Instruction::Arg VisitExpr_(const TupleNode* op) {
    Tuple tuple = GetRef<Tuple>(op);
    std::vector<Instruction::Arg> args;
    for (auto arg : tuple->fields) {
      args.push_back(this->VisitExpr(arg));
    }
    size_t dst_register = NewRegister();
    builder_->EmitCall("runtime.Tuple", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  Instruction::Arg VisitExpr_(const TupleGetItemNode* op) {
    TupleGetItem expr = GetRef<TupleGetItem>(op);
    std::vector<Instruction::Arg> args = {this->VisitExpr(expr->tuple)};

    args.push_back(builder_->ConvertConstant(expr->index));

    size_t dst_register = NewRegister();
    builder_->EmitCall("vm.builtin.tuple_getitem", args, dst_register);

    return Instruction::Arg::Register(dst_register);
  }

  String LookupFuncGlobalSymbol(Expr op) {
    if (auto* extern_func = op.as<ExternFuncNode>()) {
      return extern_func->global_symbol;
    } else if (auto* gvar = op.as<GlobalVarNode>()) {
      // Run a look up in the env to see if it maps to an extern func.
      auto it = ctx_mod_->functions.find(GetRef<GlobalVar>(gvar));
      if (it != ctx_mod_->functions.end()) {
        BaseFunc func = (*it).second;
        if (auto* efunc = func.as<ExternFuncNode>()) {
          return efunc->global_symbol;
        }
      }
      // GlobalVar can be reference to a Relax function or a TIR primfunc
      // At this point: all global var must corresponds to the right symbol.
      // TODO(relax-team): switch everything to extern before splitting TIR/relax
      // so we do not have idle global var here.
      return gvar->name_hint;
    } else {
      LOG(FATAL) << "CodeGenVM does not support calls to " << op->GetTypeKey();
      return "";
    }
  }

  void EmitAllocStorage(const Call& call_node, RegName dst_reg) {
    // Handle args of the call
    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg));
    }

    // Handle attrs of the call
    auto alloc_attrs = call_node->attrs.as<VMAllocStorageAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocStorageAttrs";
    Index runtime_device_index = alloc_attrs->runtime_device_index;
    args.push_back(builder_->ConvertConstant(runtime_device_index));
    args.push_back(builder_->ConvertConstant(alloc_attrs->dtype));

    builder_->EmitCall("vm.builtin.alloc_storage", args, dst_reg);
  }

  void EmitAllocTensor(const Call& call_node, RegName dst_reg) {
    ICHECK_EQ(call_node->args.size(), 2);
    std::vector<Instruction::Arg> args;
    args.reserve(4);
    // Handle `self`
    args.push_back(this->VisitExpr(call_node->args[0]));
    // Handle `offset`
    auto alloc_attrs = call_node->attrs.as<VMAllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocTensorAttrs";
    int offset = alloc_attrs->offset;
    args.push_back(builder_->ConvertConstant(offset));
    // Handle `shape`
    args.push_back(this->VisitExpr(call_node->args[1]));
    // Handle `dtype`
    args.push_back(builder_->ConvertConstant(alloc_attrs->dtype));

    builder_->EmitCall("vm.builtin.alloc_tensor", args, dst_reg);
  }

  void EmitCallBuiltin(const Call& call_node, RegName dst_reg) {
    auto builtin_attrs = call_node->attrs.as<BuiltinFuncAttrs>();
    ICHECK(builtin_attrs != nullptr);
    std::vector<Instruction::Arg> args;
    // if context is required, pass as first argument.
    if (builtin_attrs->require_ctx) {
      args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    }

    auto symbol = this->LookupFuncGlobalSymbol(call_node->args[0]);
    auto tuple_arg = Downcast<Tuple>(call_node->args[1]);

    // Handle args of the call
    for (Expr arg : tuple_arg->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    if (builtin_attrs->int_args.defined()) {
      for (auto val : builtin_attrs->int_args) {
        args.push_back(builder_->ConvertConstant(val->value));
      }
    }
    if (builtin_attrs->dtype_arg != DataType::Void()) {
      args.push_back(builder_->ConvertConstant(builtin_attrs->dtype_arg));
    }

    if (builtin_attrs->str_args.defined()) {
      for (auto val : builtin_attrs->str_args) {
        args.push_back(builder_->ConvertConstant(val));
      }
    }

    builder_->EmitCall(symbol, args, dst_reg);
  }

  void EmitTirDynOp(const Call& call_node, RegName dst_reg) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    auto gv = Downcast<GlobalVar>(call_node->args[0]);
    auto tir_args = Downcast<Tuple>(call_node->args[1]);
    auto func_name = gv->name_hint;

    std::vector<Instruction::Arg> args;
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));
    args.push_back(builder_->ConvertConstant(func_name));
    for (Expr arg : tir_args->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    builder_->EmitCall("vm.call_tir_dyn", args, dst_reg);
  }

  void EmitNormalCall(const Call& call_node, RegName dst_reg) {
    String name = LookupFuncGlobalSymbol(call_node->op);
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);
    builder_->EmitCall(name, args, dst_reg);
  }

  // Emit the `call_node` attributes as constants and append these constants to `args` vector.
  void AppendAttrsAsConstants(const Call& call_node, std::vector<Instruction::Arg>& args) {
    auto attrs = call_node->attrs;
    if (!attrs.defined()) return;

    if (call_node->op == unique_op_) {
      auto unique_attrs = call_node->attrs.as<UniqueAttrs>();
      args.push_back(builder_->ConvertConstant(unique_attrs->sorted));
      args.push_back(builder_->ConvertConstant(unique_attrs->return_inverse));
      args.push_back(builder_->ConvertConstant(unique_attrs->return_counts));
      args.push_back(builder_->ConvertConstant(unique_attrs->dim));
      return;
    }
    if (call_node->op == print_op_) {
      auto print_attrs = call_node->attrs.as<PrintAttrs>();
      // format string is the first argument
      args.insert(args.begin(), builder_->ConvertConstant(print_attrs->format));
      return;
    }
    if (call_node->op == assert_op_) {
      auto assert_attrs = call_node->attrs.as<AssertOpAttrs>();
      // format string comes before the format args
      args.insert(args.begin() + 1, builder_->ConvertConstant(assert_attrs->format));
      return;
    }
    LOG(FATAL) << "Support for attributes of Op " << call_node->op
               << " has not been implemented yet.";
    return;
  }

  // Emits call to packed function `name` with arguments copied over from `call_node` args and
  // attributes.
  void EmitPackedFuncCall(const Call& call_node, const FCallPacked& name, RegName dst_reg) {
    std::vector<Instruction::Arg> args = VisitArray(call_node->args);
    AppendAttrsAsConstants(call_node, args);
    builder_->EmitCall(name, args, dst_reg);
  }

  void EmitAllocClosure(const Call& call_node, RegName dst_reg) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    auto gv = Downcast<GlobalVar>(call_node->args[0]);
    auto closure_args = Downcast<Tuple>(call_node->args[1]);
    auto func_name = gv->name_hint;

    std::vector<Instruction::Arg> args;
    args.push_back(builder_->ConvertConstant(func_name));
    for (Expr arg : closure_args->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    builder_->EmitCall("vm.builtin.alloc_closure", args, dst_reg);
  }

  void EmitInvokeClosure(const Call& call_node, RegName dst_reg) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<VarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    std::vector<Instruction::Arg> args;
    // VM is utilized to help get the Function in builtin packedfunc
    args.push_back(Instruction::Arg::Register(Instruction::kVMRegister));

    args.push_back(this->VisitExpr(call_node->args[0]));

    // args for the invoke_closure
    auto invoke_closure_args = Downcast<Tuple>(call_node->args[1]);
    for (Expr arg : invoke_closure_args->fields) {
      args.push_back(this->VisitExpr(arg));
    }

    builder_->EmitCall("vm.builtin.invoke_closure", args, dst_reg);
  }

  std::vector<Instruction::Arg> VisitArray(const Array<Expr>& arr) {
    std::vector<Instruction::Arg> ret;
    for (size_t i = 0; i < arr.size(); ++i) {
      ret.push_back(this->VisitExpr(arr[i]));
    }
    return ret;
  }

  /*! \brief A counter for naming local functions. */
  size_t local_func_counter_ = 0;
  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*!
   * \brief Total number of virtual registers allocated.
   * \note The first two registers are reserved for special registers.
   */
  size_t registers_num_ = 0;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, Instruction::Arg, ObjectPtrHash, ObjectPtrEqual> var_arg_map_;
  /*! \brief the context module. */
  IRModule ctx_mod_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  const Op& alloc_storage_op_ = Op::Get("relax.vm.builtin.alloc_storage");
  const Op& alloc_tensor_op_ = Op::Get("relax.vm.builtin.alloc_tensor");
  const Op& store_shape_op_ = Op::Get("relax.vm.builtin.store_shape");
  const Op& load_shape_op_ = Op::Get("relax.vm.builtin.load_shape");
  const Op& call_tir_dyn_op_ = Op::Get("relax.vm.call_tir_dyn");
  const Op& call_builtin_op_ = Op::Get("relax.call_builtin");
  const Op& null_value_op_ = Op::Get("relax.null_value");
  const Op& unique_op_ = Op::Get("relax.unique");
  const Op& print_op_ = Op::Get("relax.print");
  const Op& assert_op_ = Op::Get("relax.assert_op");
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
};

void VMCodeGen::CodeGen(IRModule mod) {
  builder_ = relax::ExecBuilderNode::Create();
  CodeGenVM codegen(builder_, mod);
  for (auto& p : mod->functions) {
    codegen.VisitExpr(p.second);
  }
}

ObjectPtr<Executable> VMCodeGen::GetExec() { return builder_->Get(); }

/*!
 * \brief Create the Relax VM executable from an IRModule of Relax function(s) and, possibly, a
 * kernel library.
 */
Module CodeGen(IRModule mod, Target target, Optional<Module> lib, Array<Module> ext_libs,
               Map<String, runtime::NDArray> params) {
  // TODO(relax-team) Revisit the param and ext_lib options.
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
