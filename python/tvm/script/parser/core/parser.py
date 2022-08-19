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
# pylint: disable=missing-docstring
"""The core parser"""
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Set, Union
from tvm._ffi.base import TVMError

from tvm.error import DiagnosticError

from . import dispatch, doc
from .diagnostics import Diagnostics, Source
from .evaluator import eval_assign, eval_expr

DEFAULT_VISIT = {
    "Interactive",
    "Module",
    "Expression",
    "Pass",
}


def _deferred(f: Callable[[], None]):
    @contextmanager
    def context():
        try:
            yield
        finally:
            f()

    return context()


class VarTableFrame:
    vars: Set[str]

    def __init__(self):
        self.vars = set()

    def add(self, var: str):
        if var in self.vars:
            raise ValueError(f"Variable {var} already defined in current scope")
        self.vars.add(var)

    def pop_all(self, fn_pop: Callable[[str], None]):
        for var in self.vars:
            fn_pop(var)
        self.vars.clear()


class VarTable:

    frames: List[VarTableFrame]
    name2value: Dict[str, List[Any]]

    def __init__(self):
        self.frames = []
        self.name2value = defaultdict(list)

    def with_frame(self):
        def pop_frame():
            frame = self.frames.pop()
            frame.pop_all(lambda name: self.name2value[name].pop())

        self.frames.append(VarTableFrame())
        return _deferred(pop_frame)

    def add(self, var: str, value: Any):
        self.frames[-1].add(var)
        self.name2value[var].append(value)

    def get(self) -> Dict[str, Any]:
        return {key: values[-1] for key, values in self.name2value.items() if values}

    def exist(self, value: Any):
        for v in self.name2value.values():
            if v is value:
                return True
        return False


def _dispatch_wrapper(func: dispatch.ParseMethod) -> dispatch.ParseMethod:
    def _wrapper(self: "Parser", node: doc.AST) -> None:
        try:
            return func(self, node)
        except DiagnosticError:
            raise
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.report_error(node, e)
            raise

    return _wrapper


def _dispatch(self: "Parser", type_name: str) -> dispatch.ParseMethod:
    for token in [self.dispatch_tokens[-1], "default"]:
        func = dispatch.get(token=token, type_name=type_name, default=None)
        if func is not None:
            return _dispatch_wrapper(func)
    return _dispatch_wrapper(lambda self, node: self.generic_visit(node))


class Parser(doc.NodeVisitor):
    """The TVMScript parser"""

    diag: Diagnostics
    dispatch_tokens: List[str]
    var_table: VarTable

    def __init__(self, source: Source) -> None:
        self.diag = Diagnostics(source)
        self.dispatch_tokens = ["default"]
        self.var_table = VarTable()

    def parse(self, extra_vars: Optional[Dict[str, Any]] = None) -> Any:
        if extra_vars is None:
            extra_vars = {}
        with self.var_table.with_frame():
            for k, v in extra_vars.items():
                self.var_table.add(k, v)
            node = self.diag.source.as_ast()
            self.visit(node)

    def with_dispatch_token(self, token: str):
        def pop_token():
            self.dispatch_tokens.pop()

        self.dispatch_tokens.append(token)
        return _deferred(pop_token)

    def eval_expr(
        self,
        node: Union[doc.Expression, doc.expr],
        extra_vars: Optional[Dict[str, Any]] = None,
    ) -> Any:
        var_values = self.var_table.get()
        if extra_vars is not None:
            for k, v in extra_vars.items():
                var_values[k] = v
        return eval_expr(self, node, var_values)

    def _duplicate_lhs_check(self, target: doc.expr) -> Union[bool, Set[str]]:
        if isinstance(target, (doc.Tuple, doc.List)):
            vars: Set[str] = set()  # pylint: disable=redefined-builtin
            for i in target.elts:
                res = self._duplicate_lhs_check(i)
                if isinstance(res, bool) and res:
                    return True
                assert isinstance(res, set)
                if vars & res:
                    return True
                vars = vars.union(res)
            return vars
        elif isinstance(target, doc.Name):
            return {target.id}
        else:
            self.report_error(target, "Invalid type in assign statement")
            raise NotImplementedError

    def eval_assign(
        self,
        target: doc.expr,
        source: Any,
        bind_value: Callable[["Parser", doc.expr, str, Any], Any],
    ) -> Dict[str, Any]:
        if self._duplicate_lhs_check(target) is True:
            self.report_error(target, "Duplicate vars assigned.")
        var_values = eval_assign(self, target, source)
        for k, v in var_values.items():
            var = bind_value(self, target, k, v)
            self.var_table.add(k, var)
        return var_values

    def report_error(
        self, node: doc.AST, err: Union[Exception, str]
    ) -> None:  # pylint: disable=no-self-use
        if isinstance(err, TVMError):
            msg = list(filter(None, str(err).split("\n")))[-1]
        else:
            msg = str(err)
        self.diag.error(node, msg)

    def visit(self, node: doc.AST) -> None:
        if isinstance(node, (list, tuple)):
            for item in node:
                self.visit(item)
            return
        if not isinstance(node, doc.AST):
            return
        name = node.__class__.__name__.split(".")[-1]
        if name in DEFAULT_VISIT:
            func = self.generic_visit
        else:
            func = getattr(self, "visit_" + name, None)
        if func is None:
            raise NotImplementedError(f"Visitor of AST node is not implemented: {name}")
        try:
            func(node)
        except DiagnosticError:
            raise
        except Exception as e:  # pylint: disable=broad-except,invalid-name
            self.report_error(node, e)
            raise

    def visit_body(self, node: List[doc.stmt]) -> Any:
        for stmt in node:
            self.visit(stmt)

    def visit_tvm_annotation(self, node: doc.expr) -> Any:
        return _dispatch(self, "tvm_annotation")(self, node)

    def visit_FunctionDef(self, node: doc.FunctionDef) -> Any:  # pylint: disable=invalid-name
        if not node.decorator_list:
            self.report_error(node, "Function must be decorated")
        # TODO: only the last decorator is parsed
        decorator = self.eval_expr(node.decorator_list[-1])
        if not hasattr(decorator, "dispatch_token"):
            self.report_error(node, "The parser does not understand the decorator")
        token = decorator.dispatch_token
        func = dispatch.get(token=token, type_name="FunctionDef", default=None)
        if func is None:
            self.report_error(node, "The parser does not understand the decorator")
        _dispatch_wrapper(func)(self, node)

    def visit_ClassDef(self, node: doc.ClassDef) -> Any:  # pylint: disable=invalid-name
        func = dispatch.get(token="ir", type_name="ClassDef", default=None)
        if func is None:
            self.report_error(node, "The parser does not understand the decorator")
        _dispatch_wrapper(func)(self, node)

    def visit_arguments(self, node: doc.arguments) -> Any:
        return _dispatch(self, "arguments")(self, node)

    def visit_For(self, node: doc.For) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "For")(self, node)

    def visit_While(self, node: doc.While) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "While")(self, node)

    def visit_With(self, node: doc.With) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "With")(self, node)

    def visit_Assign(self, node: doc.Assign) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "Assign")(self, node)

    def visit_Expr(self, node: doc.Expr) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "Expr")(self, node)

    def visit_If(self, node: doc.If) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "If")(self, node)

    def visit_AnnAssign(self, node: doc.AnnAssign) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "AnnAssign")(self, node)

    def visit_AugAssign(self, node: doc.AugAssign) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "AugAssign")(self, node)

    def visit_Assert(self, node: doc.Assert) -> Any:
        return _dispatch(self, "Assert")(self, node)  # pylint: disable=invalid-name

    def visit_Return(self, node: doc.Return) -> Any:
        return _dispatch(self, "Return")(self, node)  # pylint: disable=invalid-name


def _handle_function(self: Parser, node: doc.FunctionDef) -> None:
    if not node.decorator_list:
        self.report_error(node, "Function must be decorated")
    # TODO: only the last decorator is parsed
    decorator = self.eval_expr(node.decorator_list[-1])
    if hasattr(decorator, "dispatch_token"):
        token = decorator.dispatch_token
        func = dispatch.get(token=token, type_name="FunctionDef", default=None)
        if func is not None:
            func(self, node)
            return
    self.report_error(node, "The parser does not understand the decorator")

    def visit_Return(self, node: doc.Return) -> Any:  # pylint: disable=invalid-name
        return _dispatch(self, "Return")(self, node)
