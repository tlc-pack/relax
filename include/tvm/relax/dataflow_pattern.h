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
 * \file tvm/relax/dataflow_pattern.h
 * \brief A pattern language for matching dataflow properties.
 */
#ifndef TVM_RELAX_DATAFLOW_PATTERN_H_
#define TVM_RELAX_DATAFLOW_PATTERN_H_

#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/relay/dataflow_pattern.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tvm {
namespace relax {

using relay::DFPattern;
using relay::DFPatternNode;

class RuntimeDepShapePattern;
/*!
 * \brief Pattern for RuntimeDepShape.
 */
class RuntimeDepShapePatternNode : public DFPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "relax.dataflow_pattern.RuntimeDepShapePattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(RuntimeDepShapePatternNode, DFPatternNode);
};

/*!
 * \brief A pattern to match expressions with runtime-dependent shapes.
 */
class RuntimeDepShapePattern : public DFPattern {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(RuntimeDepShapePattern, DFPattern, RuntimeDepShapePatternNode);
};

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_DATAFLOW_PATTERN_H_
