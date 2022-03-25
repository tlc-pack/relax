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
 * \file tvm/runtime/relax_vm/vm.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_VM_H_
#define TVM_RUNTIME_RELAX_VM_VM_H_

#include <memory>
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
  // The following fields are used for PackedFunc call within
  // a single function scope. The space is reused across multiple
  // packed func calls to increase cache locality and avoid re-allocation
  /*! \brief Temporary argument value stack for packed func call. */
  std::vector<TVMValue> call_arg_values;
  /*! \brief Temporary argument tcode stack for packed func call. */
  std::vector<int> call_arg_tcodes;

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
  /*! \brief Runtime physical device list. */
  std::vector<Device> devices;
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
   * \param frame current vm frame.
   * \param reg The register to write to.
   * \param obj The object to write to.
   */
  inline void WriteRegister(VMFrame* frame, RegName reg, const RegType& obj);
  /*!
   * \brief Read a VM register.
   * \param frame current vm frame.
   * \param reg The register to read from.
   * \return The value of the register.
   */
  inline RegType ReadRegister(VMFrame* frame, RegName reg) const;
  /*!
   * \brief Prepare function table so that func_table_[func_index] is populated.
   * \param func_index The function index.
   */
  inline void PrepareFuncTable(Index func_index);
  /*!
   * \brief Invoke a VM function.
   * \param fidx The function index.
   * \param args The arguments to the function.
   * \return The object representing the result.
   */
  RegType Invoke(Index fidx, const std::vector<RegType>& args);
  /*! \brief Run VM dispatch loop. */
  void RunLoop();
  /*!
   * \brief Run call instruction.
   * \param curr_frame The current frame.
   * \param inst The call instruction.
   */
  inline void RunInstrCall(VMFrame* curr_frame, Instruction inst);

 private:
  /*! \brief The loaded executable. */
  ObjectPtr<Executable> exec_;
  /*!
   * \brief Internal function table cache to speedup execution.
   * \note This is used to cache functions so we do not need
   *       to look up by name every time.
   *       It does mean that the definition of the function
   *       cannot change when the vm get loaded.
   */
  std::vector<PackedFunc> func_table_;
  /*!
   * \brief The current stack of call frames.
   * \note: Use unique ptr to avoid re-allocation and copy when frames_ get resized.
   */
  std::vector<std::unique_ptr<VMFrame>> frames_;
  /*! \brief The virtual machine PC. */
  Index pc_{0};
  /*! \brief The special return register. */
  RegType return_value_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_VM_H_
