/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  Sex The NOTICE file
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
 * KIND, either express or implied.  Sex The License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file unary.h
 * \brief The functions to make Relax unary arithmetic operator calls.
 */
#ifndef TVM_RELAX_OP_TENSOR_UNARY_H_
#define TVM_RELAX_OP_TENSOR_UNARY_H_

#include "../op_common.h"

namespace tvm {
namespace relax {

/*!
 * \brief Compute element-wise cos of the input data.
 * \param x The input data.
 * \return The computed result.
 */
Expr cos(Expr x);

/*! \brief Compute element-wise exp of data. */
Expr exp(Expr x);

/*! \brief Compute element-wise natural logarithm of data. */
Expr log(Expr x);

/*! \brief Compute element-wise negative value of data. */
Expr negative(Expr x);

/*! \brief Compute element-wise sigmoid of data. */
Expr sigmoid(Expr x);

/*! \brief Compute element-wise sin of data. */
Expr sin(Expr x);

/*! \brief Compute element-wise square root of data. */
Expr sqrt(Expr x);

/*! \brief Compute element-wise tanh of data. */
Expr tanh(Expr x);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TENSOR_UNARY_H_
