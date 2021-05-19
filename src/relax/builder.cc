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
 * \file src/runtime/vm/builder.ccregistry
 */

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/reflection.h>
#include <tvm/node/repr_printer.h>
#include <tvm/ir/expr.h>
#include "./bytecode.h"
#include "./executable.h"

#include <sstream>

namespace tvm {
namespace runtime {
namespace new_vm {


class Builder;

class BuilderNode : public Object {
 public:
  std::vector<Instruction> instrs;;
  std::vector<InstrArg> instr_args;
  std::unordered_map<std::string, Index> func_idx;
  std::vector<std::string> fnames;
  std::vector<ObjectRef> constants;

  void Emit(Instruction instr) {}

  void EmitCall(std::string func, std::vector<InstrArg> args, RegName ret) {
    if (func_idx.find(func) == func_idx.end()) {
      func_idx[func] = fnames.size();
      fnames.push_back(func);
    } 
    Index arg_index = instr_args.size();
    instr_args.insert(instr_args.end(), args.begin(), args.end());
    Instruction instr = Instruction::Call(func_idx[func], args.size(), arg_index, ret);
    instrs.push_back(instr);
  }

  Index AddConstant(ObjectRef obj) {
    Index idx = constants.size();
    constants.push_back(obj);
    return InstrArg(kConstIdx, idx).data;
  }

  Executable Get();

  void Print(std::ostream& os);

  TVM_DLL static Builder Create();

  void VisitAttrs(AttrVisitor* v) {
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.Builder"; 
  TVM_DECLARE_FINAL_OBJECT_INFO(BuilderNode, Object);
};

class Builder : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Builder, ObjectRef, BuilderNode);
};

Builder BuilderNode::Create() {
  Builder ret(make_object<BuilderNode>());
  return ret;
}

Executable BuilderNode::Get() {
  Bytecode code;
  for (const Instruction& instr : this->instrs) {
    code.text.push_back(static_cast<int64_t>(instr.op));
    code.text.push_back(instr.dst);
    switch(instr.op) {
      case Opcode::Call: {
        code.text.push_back(instr.func_index);
        code.text.push_back(instr.num_args);
        code.text.push_back(instr.arg_index);
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
  for (const InstrArg& arg : this->instr_args) {
    code.data.push_back(arg.data); 
  }
  return ExecutableNode::Create(code);
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

std::string InstrArgToStr(InstrArg arg) {
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
  for (const Instruction& instr : instrs) {
    switch (instr.op) {
      case Opcode::Call: {
        os << "call " << this->fnames[instr.func_index] << " \tin: "
           << StrJoin<InstrArg>(this->instr_args.data(), instr.arg_index, instr.num_args, ", ", InstrArgToStr)
           << " \tret: " << RegNameToStr(instr.dst) << "\n";
        break;
      }
      default:
        LOG(FATAL) << "should never hit this case: " << static_cast<int>(instr.op);
        break;
    }
  }
}

TVM_REGISTER_NODE_TYPE(BuilderNode);

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


}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
