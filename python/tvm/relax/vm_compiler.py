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
# pylint: disable=no-else-return, unidiomatic-typecheck, undefined-variable, invalid-name, redefined-builtin
"""
The Relax Virtual Machine compiler.
"""
from typing import List, Optional, Union, Dict
import tvm
from . import vm, _ffi_api


def compile(mod: tvm.IRModule) -> vm.Executable:
    """Compile the module to VM executable. A helper function for VMCompiler.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relay module to build.

    Returns
    -------
    exec : tvm.relax.Executable
        The VM executable that contains the bytecode.
    """
    compiler = VMCompiler()
    compiler.compile(mod)
    return compiler.get_exec()


class VMCompiler(object):
    """Compiler that compiles module to VM executable."""

    def __init__(self):
        self.mod = _ffi_api.VMCompiler()
        self._compile = self.mod["compile"]
        self._get_exec = self.mod["get_executable"]

    def compile(self, mod: tvm.IRModule) -> None:
        """Compile the module to VM executable.

        Parameters
        ----------
        mod : tvm.IRModule
            The IRModule to build.
        """
        self._compile(mod)

    def get_exec(self) -> vm.Executable:
        """Get the VM executable.

        Returns
        -------
        exec : tvm.relax.Executable
            The VM executable that contains bytecode.
        """
        return self._get_exec()
