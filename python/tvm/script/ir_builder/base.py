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
"""A generic IRBuilder across the TVM stack"""
from typing import List, TypeVar

from tvm._ffi import register_object as _register_object
from tvm.runtime import Object as _Object

from . import _ffi_api


@_register_object("script.ir_builder.IRBuilderFrame")
class IRBuilderFrame(_Object):
    def __enter__(self) -> "IRBuilderFrame":
        _ffi_api.IRBuilderFrameEnter(self)  # pylint: disable=no-member # type: ignore
        return self

    def __exit__(self, ptype, value, trace) -> None:  # pylint: disable=unused-argument
        _ffi_api.IRBuilderFrameExit(self)  # pylint: disable=no-member # type: ignore

    def add_callback(self, callback) -> None:  # pylint: disable=unused-argument
        _ffi_api.IRBuilderFrameAddCallback(  # pylint: disable=no-member # type: ignore
            self, callback
        )


@_register_object("script.ir_builder.IRBuilder")
class IRBuilder(_Object):
    def __init__(self) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.IRBuilder  # pylint: disable=no-member # type: ignore
        )

    def __enter__(self) -> "IRBuilder":
        _ffi_api.IRBuilderEnter(self)  # pylint: disable=no-member # type: ignore
        return self

    def __exit__(self, ptype, value, trace) -> None:  # pylint: disable=unused-argument
        _ffi_api.IRBuilderExit(self)  # pylint: disable=no-member # type: ignore

    @staticmethod
    def current() -> "IRBuilder":
        return _ffi_api.IRBuilderCurrent()  # pylint: disable=no-member # type: ignore

    def get(self) -> _Object:
        return _ffi_api.IRBuilderGet(self)  # pylint: disable=no-member # type: ignore


DefType = TypeVar("DefType", bound=_Object)


def name(s: str, v: DefType) -> DefType:
    return _ffi_api.IRBuilderName(s, v)  # pylint: disable=no-member # type: ignore


def name_many(  # pylint: disable=invalid-name
    s: List[str],
    vs: List[DefType],
) -> List[DefType]:
    assert len(s) == len(vs)
    return [name(i, v) for i, v in zip(s, vs)]
