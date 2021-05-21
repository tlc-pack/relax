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

Executable BuilderNode::Get() {
  return Executable(this->exec);
}

void DLDatatypePrint(std::ostream& os, const DLDataType& dtype) {
  switch (dtype.code) {
    case kDLInt:
      os << "int";
      break;
    case kDLUInt:
      os << "uint";
      break;
    case kDLFloat:
      os << "float";
      break;
  }

  os << int(dtype.bits);
  if (dtype.lanes != 1) {
    os << "x" << dtype.lanes;
  }
}

template <typename T>
std::string StrJoin(T* items, int offset, int cnt,
                    std::string delim = ", ",
                    std::function<std::string(T)> repr = std::to_string) {
  if (cnt == 0) {
    return "";
  }
  std::ostringstream oss;
  oss << repr(items[offset]);
  for (int i = 1; i < cnt; ++i) {
    oss << delim << repr(items[offset + i]);
  }
  return oss.str();
}

std::string RegNameToStr(RegName reg) {
  if (reg == kVoidArg) {
    return "void";
  } else {
    return "%" + std::to_string(reg);
  }

}

std::string ExecWordToStr(ExecWord word) {
  // only for argument
  InstrArg arg(word);
  switch(arg.kind()) {
    case kRegister:
      return RegNameToStr(arg.value());
    case kImmediate:
      return "i" + std::to_string(arg.value());
    case kConstIdx:
      return "c[" + std::to_string(arg.value()) + "]";
    default:
      LOG(FATAL) << "Wrong instruction kind: " << arg.kind();
      return "";
  }
}

void BuilderNode::Print(std::ostream& os) {
  // print the text format
  for (size_t i = 0; i < exec->instr_offset.size(); ++i) {
    Instruction instr = exec->GetInstruction(i);
    switch (instr.op) {
      case Opcode::Call: {
        os << "call " << exec->func_names[instr.func_idx] << " \tin: "
           << StrJoin<ExecWord>(instr.args, 0, instr.num_args, ", ", ExecWordToStr)
           << " \tret: " << RegNameToStr(instr.dst) << "\n";
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
}

TVM_REGISTER_GLOBAL("relax.BuilderCreate").set_body_typed(BuilderNode::Create);

TVM_REGISTER_GLOBAL("relax.BuilderEmitCall").set_body_typed(
[](Builder builder, String name, Array<IntImm> args, int64_t ret) {
  std::vector<InstrArg> args_;
  for (size_t i = 0; i < args.size(); ++i) {
    args_.push_back(static_cast<InstrArg>(args[i]->value));
  }
  InstrArg ret_(ret);
  CHECK_EQ(ret_.kind(), ArgKind::kRegister);
  builder->EmitCall(name, args_, ret_.value());
});

TVM_REGISTER_GLOBAL("relax.BuilderAddConstant").set_body_typed([](Builder builder, ObjectRef obj) {
  return builder->AddConstant(obj);
});

TVM_REGISTER_GLOBAL("relax.BuilderGet").set_body_typed([](Builder builder) {
  return builder->Get();
});

TVM_REGISTER_GLOBAL("relax.BuilderPrint").set_body_typed([](Builder builder) {
  builder->Print(LOG(INFO) << "\n");
});


}  // namespace relax
}  // namespace tvm
