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

import tvm
from tvm.runtime import Object
from tvm._ffi.base import _LIB, check_call
from . import _ffi_api


@tvm._ffi.register_object("relax.Executable")
class Executable(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Executable)

    def stats(self):
        return _ffi_api.ExecutableStats(self)

    def save_to_file(self, file_name):
        return _ffi_api.ExecutableSaveToFile(self, file_name)

    def astext(self):
        return _ffi_api.ExecutableAsText(self)
    
    def aspython(self):
        return _ffi_api.ExecutableAsPython(self)

def load_exec_from_file(file_name):
    return _ffi_api.ExecutableLoadFromFile(file_name)

class VirtualMachine(object):
    def __init__(self, exec, mod=None):
        self.module = _ffi_api.VirtualMachine(exec, mod)
        self._run = self.module["run"]

    def run(self):
        return self._run()
