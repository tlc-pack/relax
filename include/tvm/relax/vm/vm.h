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
 * \file tvm/relax/vm/vm.h
 * \brief
 */
#ifndef TVM_RELAX_VM_VM_H_
#define TVM_RELAX_VM_VM_H_

#include <string>
#include <vector>

#include "./bytecode.h"
#include "./executable.h"
#include "./memory_manager.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief The register type.
 */
using RegType = TVMRetValue;

/*!
 * \brief A representation of a stack frame.
 *
 * A stack frame is a record containing the information needed
 * to restore the caller's virtual machine state after returning
 * from a function call.
 */
struct VMFrame {
  /*! \brief The return program counter. */
  Index return_pc;
  /*! \brief Statically allocated space for objects */
  std::vector<RegType> register_file;
  /*! \brief Register in caller's frame to put return value */
  RegName caller_return_register;

  VMFrame(Index pc, Index register_file_size)
      : return_pc(pc), register_file(register_file_size), caller_return_register(0) {}
};

/*!
 * \brief The state of virtual machine, which can be referred in
 * instruction.
 */
struct VMState {
  /*! \brief The memory allocators. */
  std::vector<Allocator*> allocators;
  /*! \brief The kernel library. */
  Optional<runtime::Module> lib;
};

/*!
 * \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the executable.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialize them to disk or over the
 * wire.
 */
class VirtualMachine : public runtime::ModuleNode {
 public:
  /*!
   * \brief Initialize the virtual machine for a set of devices.
   * \param devices The set of TVM devices.
   * \param alloc_types The allocator types for each device.
   */
  void Init(const std::vector<Device>& devices, const std::vector<AllocatorType>& alloc_types);
  /*!
   * \brief Load the executable for the virtual machine.
   * \param exec The executable.
   */
  void LoadExecutable(ObjectPtr<Executable> exec);
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function needs resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  ~VirtualMachine() final {}

  const char* type_key() const final { return "relax.VirtualMachine"; }

  /*! \brief The state of the virtual machine, which can be referred by instructions. */
  VMState state;

 protected:
  /*!
   * \brief Push a call frame on to the call stack.
   * \param ret_pc The program counter to return to.
   * \param vm_func The function to be pushed to the call stack.
   */
  void PushFrame(Index ret_pc, const VMFunction& vm_func);
  /*!
   * \brief Pop a frame off the call stack.
   */
  void PopFrame();
  /*!
   * \brief Write to a VM register.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  inline void WriteRegister(RegName reg, const RegType& obj);
  /*!
   * \brief Read a VM register.
   * \param reg The register to read from.
   * \return The value of the register.
   */
  inline RegType ReadRegister(RegName reg) const;
  /*!
   * \brief Invoke a VM function.
   * \param fidx The function index.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  RegType Invoke(Index fidx, const std::vector<RegType>& args);
  /*! \brief Run VM dispatch loop. */
  void RunLoop();

 private:
  /*! \brief The loaded executable. */
  ObjectPtr<Executable> exec_;
  /*! \brief The current stack of call frames. */
  std::vector<VMFrame> frames_;
  /*! \brief The virtual machine PC. */
  Index pc_{0};
  /*! \brief The special return register. */
  RegType return_value_;
  /*! \brief The devices. */
  std::vector<Device> devices_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RELAX_VM_VM_H_
