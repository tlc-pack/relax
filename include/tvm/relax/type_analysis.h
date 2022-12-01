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
 * \file tvm/relax/type_analysis.h
 * \brief Relax type analysis APIs.
 */
#ifndef TVM_RELAX_TYPE_ANALYSIS_H_
#define TVM_RELAX_TYPE_ANALYSIS_H_

#include <tvm/ir/type.h>

namespace tvm {
namespace relax {
/*!
 * \brief Check the subtype relationship between base and derived.
 * \param base The base type.
 * \param derived The derived type.
 * \return If \p derived is a subtype of \p base or if both are the same type, returns true.
 * Otherwise returns false.
 */
bool IsBaseOf(const Type& base, const Type& derived);


/*!
 * \brief Find the lowest common ancestor of two types.
 * \param t Type 1.
 * \param u Type 2.
 * \return The lowest common ancestor of two types.
 */
Type FindLCA(const Type& t1, const Type& t2);

}  // namespace relax
}  // namespace tvm
#endif  // TVM_RELAX_TYPE_ANALYSIS_H_
