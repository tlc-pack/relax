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
 * \file src/relax/vm/executable.cc
 * \brief 
 */

#include <tvm/runtime/logging.h>
#include <tvm/relax/vm/executable.h>
#include <functional>

#include <sstream>

namespace tvm {
namespace relax {
namespace vm {

TVM_REGISTER_NODE_TYPE(ExecutableNode);

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
  if (reg == Instruction::kVoidArg) {
    return "void";
  } else {
    return "%" + std::to_string(reg);
  }
}

std::string InstrArgToStr(InstrArg arg) {
  // only for argument
  switch(arg.kind()) {
    case Instruction::kRegister:
      return RegNameToStr(arg.value());
    case Instruction::kImmediate:
      return "i" + std::to_string(arg.value());
    case Instruction::kConstIdx:
      return "c[" + std::to_string(arg.value()) + "]";
    default:
      LOG(FATAL) << "Wrong instruction kind: " << arg.kind();
      return "";
  }
}

String ExecutableNode::AsText() const {
  // print the text format
  std::ostringstream os;
  for (size_t i = 0; i < this->instr_offset.size(); ++i) {
    Instruction instr = this->GetInstruction(i);
    switch (instr.op) {
      case Opcode::Call: {
        os << "call " << this->func_names[instr.func_idx] << " \tin: "
           << StrJoin<InstrArg>(instr.args, 0, instr.num_args, ", ", InstrArgToStr)
           << " \tret: " << RegNameToStr(instr.dst) << "\n";
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
  return String(os.str());
}
TVM_REGISTER_GLOBAL("relax.Executable").set_body_typed([]() {
  return Executable();
});

TVM_REGISTER_GLOBAL("relax.ExecutableAsText").set_body_typed([](Executable exec) {
  return exec->AsText();
});


}  // namespace vm
}  // namespace relax
}  // namespace tvm
