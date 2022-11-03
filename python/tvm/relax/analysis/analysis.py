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
"""
This file contains the set of passes for Relax, which exposes an interface for
configuring the passes and scripting them in Python.
"""

from typing import Dict, List

import tvm
from tvm import tir
from tvm.relax.expr import DataflowBlock, GlobalVar, Var, Expr, Function, Binding
from tvm.ir.module import IRModule
from . import _ffi_api


def post_order_visit(expr, fvisit):
    """Recursively visit the ir in post DFS order node,
    apply fvisit. Each node is guaranteed to be visited
    only once.

    Parameters
    ----------
    expr : tvm.relay.Expr
        The input expression.

    fvisit : function
        The visitor function to be applied.
    """
    return _ffi_api.post_order_visit(expr, fvisit)


def well_formed(mod: tvm.IRModule) -> bool:
    """Check if the IRModule is well formed.

    Parameters
    ----------
    mod : tvm.IRModule
        The input IRModule.

    Returns
    -------
    ret: bool
        True if the IRModule is well formed, False if not.
    """
    return _ffi_api.well_formed(mod)


def get_var2val(func: Function) -> Dict[Var, Expr]:
    """
    Get a mapping from Var to Expr for each variable in the function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Returns
    -------
    Dict[Var, Expr]
        A mapping from Var to Expr.
    """
    return _ffi_api.get_var2val(func)


def udchain(dfb: DataflowBlock) -> Dict[Var, List[Var]]:
    """
    Analyze the variable use-def chain in a dataflow block.

    Parameters
    ----------
    dfb : DataflowBlock
        The dataflow block to analyze

    Returns
    -------
    Dict[Var, List[Var]]
        A mapping from variable definition to its uses.
    """
    return _ffi_api.udchain(dfb)


def name_to_binding(func: Function) -> Dict[str, List[Binding]]:
    """Return a map from variable name to its bindings."""
    return _ffi_api.name_to_binding(func)


def remove_all_unused(func: Function) -> Function:
    """Remove all unused variables from the function.

    Parameters
    ----------
    func : Function
        The input function to be analyzed.

    Returns
    -------
    Function
        The function with unused variables removed.
    """
    return _ffi_api.remove_all_unused(func)


def shape_vars(expr: Expr) -> List[tir.Var]:
    """
    Returns all shape variables (TIR variables) in the given expression.

    Note that the expression is intended to be a shape expression, i.e.,
    one used as the `shape_` for another expression.

    Parameters
    ----------
    expr : Expr
        The expression. Meant to be a shape expression.

    Returns
    -------
    ret: List[tir.Var]
        A list of all shape variables (TIR variables) in the expression.
    """
    return _ffi_api.shape_vars(expr)


def derive_func_ret_shape(args: List[Var], body: Expr) -> Expr:
    """
    Given the argument vars and body, derives a return shape for
    a function with those args and that body.
    If the body's shape contains free shape vars (those not used in the args), the
    return shape is relaxed to RuntimeDepShape; otherwise, the body's shape is used.

    Parameters
    ----------
    args: List[Var]
        The argument variables, ideally with the shape_ field filled in

    body: Expr
        The functino body, ideally with the shape_ field filled in

    Returns
    -------
    ret: Expr
        An expression that can serve as the return shape for the function
    """
    return _ffi_api.derive_func_ret_shape(args, body)


def bound_vars(expr: Expr) -> List[Var]:
    """
    Return all bound variables from expression expr.

    Bound variables are all variables that are declared in the expr.
    They only have meaning inside that expr, and can only be used in it.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of bound vars in expr, in post-DFS order
    """
    return _ffi_api.bound_vars(expr)


def free_vars(expr: Expr) -> List[Var]:
    """
    Return all free variables from expression expr.

    Free variables are variables that are not bound by a
    VarBinding or a function parameter in the expression.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of free vars in expr, in post-DFS order
    """
    return _ffi_api.free_vars(expr)


def all_vars(expr: Expr) -> List[Var]:
    """
    Return all (local) variables from expression expr.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[Var]
        List of vars in expr, in post-DFS order
    """
    return _ffi_api.all_vars(expr)


def all_global_vars(expr: Expr) -> List[GlobalVar]:
    """
    Return all global variables from expression expr.

    Parameters
    ----------
    expr: Expr
        The expression.

    Returns
    -------
    ret: List[GlobalVar]
        List of global vars in expr, in post-DFS order
    """
    return _ffi_api.all_global_vars(expr)


def called_global_vars(expr: Expr) -> List[GlobalVar]:
    """
    Return all global vars called (potentially recursively) from expr.

    Parameters
    ----------
    expr: Expr
        The expression

    Returns
    -------
    ret: List[GlobalVar]
        List of global vars that are used recursively in expr,
        in post-DFS order
    """
    return _ffi_api.called_global_vars(expr)


def extract_buffer_info(main_func: Function, mod: IRModule):
    """This analysis pass consumes an IRModule with a Relax main function
        that defines a ordering in the callees to operators and produces BufferInfo
        objects that contains information about tensor allocations and liveness
        conflicts between allocations.

    Parameters
    ----------
    main_func: tvm.relax.Function
        The main function containing calls to operator PrimFuncs.
    mod : tvm.ir.IRModule
        The full IRModule containing all Relax and Prim functions.

    Returns
    -------
    Map<relax::Expr, BufferInfo>
        extracted buffer info objects
    """
    return _ffi_api.extract_buffer_info(main_func, mod)
