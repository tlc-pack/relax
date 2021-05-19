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
 * \file src/runtime/vm/executable.cc
 * \brief 
 */

#include <tvm/runtime/logging.h>
#include <functional>
#include "./executable.h"

#include <sstream>

namespace tvm {
namespace runtime {
namespace new_vm {

Executable ExecutableNode::Create(Bytecode code) {
  Executable ret(make_object<ExecutableNode>());
  ret->code = code;
  return ret;
}

TVM_REGISTER_NODE_TYPE(ExecutableNode);


}  // namespace new_vm
}  // namespace runtime
}  // namespace tvm
