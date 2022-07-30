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
from ...ir_builder import ir as I
from .._core import Parser, dispatch, doc


@dispatch.register(token="ir", type_name="ClassDef")
def _visit_class_def(self: Parser, node: doc.ClassDef) -> None:
    with self.var_table.with_frame():
        with I.ir_module():
            for stmt in node.body:
                if isinstance(stmt, doc.FunctionDef):
                    gv = I.add_function(stmt.name, func=None)
                    self.var_table.add(stmt.name, gv)
            with self.with_dispatch_token("ir"):
                self.visit_body(node.body)


@dispatch.register(token="ir", type_name="Assign")
def _visit_assign(_self: Parser, _node: doc.Assign) -> None:
    pass


@dispatch.register(token="ir", type_name="Expr")
def _visit_expr(_self: Parser, _node: doc.Expr) -> None:
    pass
