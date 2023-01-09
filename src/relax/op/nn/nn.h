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
 * \file nn.h
 * \brief The functions to make Relax neural network operator calls.
 */

#ifndef TVM_RELAX_OP_NN_NN_H_
#define TVM_RELAX_OP_NN_NN_H_

#include <tvm/relax/attrs/nn.h>

#include "../op_common.h"

namespace tvm {
namespace relax {

/*! \brief Rectified linear unit. */
Expr relu(Expr data);

/*! \brief Gaussian Error Linear Units function. */
Expr gelu(Expr data);

/*! \brief Sigmoid Linear Unit function. */
Expr silu(Expr data);

/*! \brief Softmax function. */
Expr softmax(Expr data, int axis);

/*! \brief Compute batch normalization. */
Expr batch_norm(Expr data, Expr gamma, Expr beta, Expr moving_mean, Expr moving_var,  //
                int axis, double epsilon, bool center, bool scale);

/*! \brief Compute layer normalization. */
Expr layer_norm(Expr data, Expr gamma, Expr beta, Array<Integer> axes, double epsilon, bool center,
                bool scale);

/*!
 * \brief Applies the dropout operation to the input tensor.
 * \param data The input data to the operator.
 * \param rate The probability for an element to be reset to 0.
 * \return A Tuple of two tensors.
 * The first one is the original tensor and the second one is a
 * mask tensor (1.0 where element not dropped, 0.0 where dropped)
 */
Expr dropout(Expr data, double rate);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_NN_NN_H_
