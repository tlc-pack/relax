# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=no-else-return
# pylint: disable=unidiomatic-typecheck
from . import _ffi_api

def fma_rewrite(expr):
    """Perform fused multiply add rewriting in dataflow blocks.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    """
    return _ffi_api.fma_rewrite(expr)

def explicit_memory_rewrite(expr):
    """Perform explicit memory allocation for call_dps in dataflow blocks.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    """
    return _ffi_api.explicit_memory_rewrite(expr)

def shape_lower(expr):
    """Perform explicit memory allocation for call_dps in dataflow blocks.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.
    """
    return _ffi_api.shape_lower(expr)
