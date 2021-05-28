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
 * \file src/relax/builder.cc
 */

#include <tvm/relax/builder.h>
#include <sstream>

namespace tvm {
namespace relax {

using namespace vm;

TVM_REGISTER_NODE_TYPE(BuilderNode);


Builder BuilderNode::Create() {
  Builder ret(make_object<BuilderNode>());
  ret->exec = make_object<ExecutableNode>();
  return ret;
}

vm::Index BuilderNode::EmitConstant(ObjectRef obj) {
  vm::Index idx = exec->constants.size();
  exec->constants.push_back(obj);
  return vm::InstrArg(vm::Instruction::kConstIdx, idx).data;
}

void BuilderNode::Function(std::string func_name, int64_t num_inputs) {
  const auto& m = exec->global_map;
  ICHECK(m.find(func_name) == m.end());
  VMFunction vmfunc;
  vmfunc.name = func_name;
  vmfunc.start_instr = exec->instr_offset.size();
  vmfunc.num_args = num_inputs;
  exec->global_map[func_name] = exec->global_funcs.size();
  exec->global_funcs.push_back(vmfunc);
}

void BuilderNode::EmitCall(std::string func, std::vector<InstrArg> args, RegName dst) {
  // store function
  if (exec->func2idx.find(func) == exec->func2idx.end()) {
    exec->func2idx[func] = exec->func_names.size();
    exec->func_names.push_back(func);
  } 
  Index func_idx = exec->func2idx[func];
  // store instruction
  exec->instr_offset.push_back(exec->instr_data.size());
  exec->instr_data.push_back(static_cast<ExecWord>(Opcode::Call));
  exec->instr_data.push_back(dst);
  exec->instr_data.push_back(func_idx);
  exec->instr_data.push_back(args.size());
  // store arguments
  std::transform(args.cbegin(), args.cend(),
                 std::back_inserter(exec->instr_data),
                 [](InstrArg arg){ return arg.data; });
}

void BuilderNode::EmitRet(RegName result) {
  exec->instr_offset.push_back(exec->instr_data.size());
  exec->instr_data.push_back(static_cast<ExecWord>(Opcode::Ret));
  exec->instr_data.push_back(result);
}

Executable BuilderNode::Get() {
  return Executable(this->exec);
}

bool BuilderNode::Check() {
  // check if registers are used correctly
  const VMFunction& gfunc = this->exec->global_funcs.back();
  Index num_inputs = gfunc.num_args;
  std::unordered_set<RegName> dst_registers;
  std::unordered_set<RegName> arg_registers;
  size_t start_instr = gfunc.start_instr;
  size_t end_instr = this->exec->instr_offset.size();
  for (size_t idx = start_instr; idx < end_instr; ++idx) {
    Instruction instr = this->exec->GetInstruction(idx);
    switch (instr.op) {
      case Opcode::Call: {
        for (int i = 0; i < instr.num_args; ++i) {
          if (instr.args[i].kind() == Instruction::kRegister && 
              instr.args[i].value() >= num_inputs && 
              dst_registers.find(instr.args[i].value()) == dst_registers.end()) {
            LOG(ERROR) << "register r(" << instr.args[i].value()
                       << ") in VM function \"" << gfunc.name
                       << "\" is used as input while the number of inputs is only " << num_inputs << ".\n";
            return false;
          }
          arg_registers.emplace(instr.args[i].value());
        }
        if (instr.dst != Instruction::kVoidArg) {
          dst_registers.emplace(instr.dst);
        }
        break;
      }
      case Opcode::Ret: {
        arg_registers.emplace(instr.result);
        for (int i = 0; i < num_inputs; i++) {
          if (arg_registers.find(i) == arg_registers.end()) {
            LOG(WARNING) << "register r(" << i << ") in VM function \"" << gfunc.name << "\" is unused as input.\n";
          }
        }
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
  return true;
}

void BuilderNode::Formalize() {
  // a pass to formalize user-specified register indexes in the order of use
  // and decide the number of registers to allocate for a VMFunction
  const VMFunction& gfunc = this->exec->global_funcs.back();
  Index num_inputs = gfunc.num_args;
  RegName register_idx = num_inputs;
  std::unordered_map<RegName, RegName> register_map;
  size_t start_instr = gfunc.start_instr;
  size_t end_instr = this->exec->instr_offset.size();
  for (size_t idx = start_instr; idx < end_instr; ++idx) {
    Instruction instr = this->exec->GetInstruction(idx);
    switch (instr.op) {
      case Opcode::Call: {
        for (int i = 0; i < instr.num_args; ++i) {
          if (instr.args[i].kind() == Instruction::kRegister && 
              register_map.find(instr.args[i].value()) != register_map.end()) {
            this->exec->instr_data[this->exec->instr_offset[idx] + 4 + i] = register_map[instr.args[i].value()];
          }
        }
        if (instr.dst != Instruction::kVoidArg && instr.dst >= num_inputs && 
            register_map.find(instr.dst) == register_map.end()) {
          this->exec->instr_data[this->exec->instr_offset[idx] + 1] = register_idx;
          register_map[instr.dst] = register_idx++;
        }
        break;
      }
      case Opcode::Ret: {
        if (register_map.find(instr.result) != register_map.end()) {
          this->exec->instr_data[this->exec->instr_offset[idx] + 1] = register_map[instr.result];
        }
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
  this->exec->global_funcs.back().register_file_size = register_idx;
}

TVM_REGISTER_GLOBAL("relax.BuilderCreate")
.set_body_typed(BuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.BuilderEmitConstant")
.set_body_typed([](Builder builder, ObjectRef obj) {
  return builder->EmitConstant(obj);
});

TVM_REGISTER_GLOBAL("relax.BuilderFunction")
.set_body_typed([](Builder builder, String name, int64_t num_inputs) {
  return builder->Function(name, num_inputs);
});

TVM_REGISTER_GLOBAL("relax.BuilderEmitCall")
.set_body_typed([](Builder builder, String name,
                   Array<IntImm> args, int64_t dst) {
  std::vector<InstrArg> args_;
  for (size_t i = 0; i < args.size(); ++i) {
    args_.push_back(static_cast<InstrArg>(args[i]->value));
  }
  InstrArg dst_(dst);
  CHECK_EQ(dst_.kind(), Instruction::ArgKind::kRegister);
  builder->EmitCall(name, args_, dst_.value());
});

TVM_REGISTER_GLOBAL("relax.BuilderEmitRet")
.set_body_typed([](Builder builder, int64_t result) {
  builder->EmitRet(result);
});

TVM_REGISTER_GLOBAL("relax.BuilderR")
.set_body_typed([](Builder builder, int64_t value) {
  return InstrArg(Instruction::kRegister, value).data;
});

TVM_REGISTER_GLOBAL("relax.BuilderImm")
.set_body_typed([](Builder builder, int64_t value) {
  return InstrArg(Instruction::kImmediate, value).data;
});

TVM_REGISTER_GLOBAL("relax.BuilderC")
.set_body_typed([](Builder builder, int64_t value) {
  return InstrArg(Instruction::kConstIdx, value).data;
});

TVM_REGISTER_GLOBAL("relax.BuilderGet")
.set_body_typed([](Builder builder) {
  return builder->Get();
});

TVM_REGISTER_GLOBAL("relax.BuilderCheck")
.set_body_typed([](Builder builder) {
  return builder->Check();
});

TVM_REGISTER_GLOBAL("relax.BuilderFormalize")
.set_body_typed([](Builder builder) {
  return builder->Formalize();
});

}  // namespace relax
}  // namespace tvm
