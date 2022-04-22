/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_BINARY_BROADCAST_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

RELAX_REGISTER_BINARY_BROADCAST_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1);

RELAY_TO_RELAX_BINARY_ATTRS("multiply");
RELAY_TO_RELAX_BINARY_ATTRS("subtract");

RELAY_REGISTER_OP("collapse_sum_like")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapeBinaryLike)            \
      .set_attr<FInferType>("FInferType", InferTypeBinaryLike);

RELAY_REGISTER_OP("nn.dense")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapeBinaryNNDense)            \
      .set_attr<FInferType>("FInferType", InferTypeBinaryNNDense);

RELAY_REGISTER_OP("transpose")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapeBinaryTranspose)            \
      .set_attr<FInferType>("FInferType", InferTypeBinaryTranspose);

RELAY_REGISTER_OP("nn.relu")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);

RELAY_REGISTER_OP("zeros_like")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);

RELAY_REGISTER_OP("nn.log_softmax")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);
RELAY_REGISTER_OP("exp")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);
RELAY_REGISTER_OP("log")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);
RELAY_REGISTER_OP("negative")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);

RELAY_REGISTER_OP("ones_like")                                       \
      .set_attr<FInferShape>("FInferShape", InferShapePWUnary)            \
      .set_attr<FInferType>("FInferType", InferTypePWUnary);

RELAY_REGISTER_OP("where") \
      .set_attr<FInferShape>("FInferShape", InferShapeWhere)            \
      .set_attr<FInferType>("FInferType", InferTypeWhere);

RELAY_REGISTER_OP("less") \
      .set_attr<FInferShape>("FInferShape", InferShapeCmp)            \
      .set_attr<FInferType>("FInferType", InferTypeCmp);

RELAY_REGISTER_OP("sum") \
      .set_attr<FInferShape>("FInferShape", InferShapeReduce)            \
      .set_attr<FInferType>("FInferType", InferTypeReduce);

RELAY_REGISTER_OP("nn.cross_entropy") \
      .set_attr<FInferShape>("FInferShape", InferShapeCrossEntropy)            \
      .set_attr<FInferType>("FInferType", InferTypeCrossEntropy);

RELAX_REGISTER_BINARY_BROADCAST_OP("divide")
    .describe("Elementwise divide with broadcasting")
    .set_support_level(1);

}  // namespace relax
}  // namespace tvm
