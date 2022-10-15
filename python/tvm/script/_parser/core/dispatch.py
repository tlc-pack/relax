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
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Type

from .doc import AST

if TYPE_CHECKING:
    from .parser import Parser


ParseMethod = Callable[["Parser", AST], None]
ParseVTable: Dict[Tuple[str, str], ParseMethod] = {}

OpMethod = Callable[..., Any]
OpVTable: Dict[Tuple[Type, AST, int], OpMethod] = {}


def register(token: str, type_name: str):
    """Register a method for a dispatch token and type name"""

    def f(method: ParseMethod):
        ParseVTable[(token, type_name)] = method

    return f


def get(
    token: str,
    type_name: str,
    default: Optional[ParseMethod] = None,
) -> Optional[ParseMethod]:
    return ParseVTable.get((token, type_name), default)


def register_op(ty: Type, op: AST, operand_index: int):  # pylint: disable=invalid-name
    def f(method: OpMethod):
        OpVTable[(ty, op, operand_index)] = method

    return f


def get_op(  # pylint: disable=invalid-name
    ty: Type,
    op: Type,
    operand_index: int,
    default: Optional[OpMethod] = None,
) -> Optional[OpMethod]:
    return OpVTable.get((ty, op, operand_index), default)
