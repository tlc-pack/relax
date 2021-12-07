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

from typing import List, Optional, Union, Dict, Tuple
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.runtime import Object, Device, Module, PackedFunc
from tvm._ffi.base import _LIB, check_call
from tvm.tir.function import PrimFunc
from . import _ffi_api
from . import transform
from ..rpc.base import RPC_SESS_MASK


@tvm._ffi.register_object("relax.Executable")
class Executable(Object):
    """The executable object emitted by the VM compiler or the ExecBuilder."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Executable)

    def stats(self) -> str:
        """print the detailed statistics of the executable."""
        return _ffi_api.ExecutableStats(self)

    def save_to_file(self, file_name: str) -> None:
        """serialize and write the executable to a file."""
        _ffi_api.ExecutableSaveToFile(self, file_name)

    def astext(self) -> str:
        """print the instructions as text format."""
        return _ffi_api.ExecutableAsText(self)

    def aspython(self) -> str:
        """print the instructions as python program."""
        return _ffi_api.ExecutableAsPython(self)


def load_exec_from_file(file_name: str) -> Executable:
    return _ffi_api.ExecutableLoadFromFile(file_name)


class VirtualMachine(object):
    """Relax VM runtime."""

    NAIVE_ALLOCATOR = 1
    POOLED_ALLOCATOR = 2

    def __init__(
        self,
        exec: Executable,
        device: Union[Device, List[Device]],
        memory_cfg: Optional[Union[str, Dict[Device, str]]] = None,
        mod: Optional[Module] = None,
    ) -> None:
        """
        Construct a VirtualMachine wrapper object.

        Parameters
        ----------
        exec: Executable
            The VM executable.

        device : tvm.runtime.Device or List[tvm.runtime.Device]
            The device to deploy the module.

        memory_cfg : str or Dict[tvm.runtime.Device, str], optional
            Config the type of memory allocator. The allocator type can be ["naive",
            "pooled"]. If memory_cfg is None, all devices will use pooled allocator
            by default. If memory_cfg is string, all devices will use the specified
            allocator type. If memory_cfg is a dict, each device uses the allocator
            type specified in the dict, or pooled allocator if not specified in the
            dict.

        mod : tvm.runtime.Module, optional
            Optional runtime module to load to the VM.

        Returns
        -------
        vm: VirtualMachine
            A VM wrapper object.
        """
        self.module = _ffi_api.VirtualMachine(exec, mod)
        self._setup_device(device, memory_cfg)

    def _setup_device(self, dev: Device, memory_cfg: Union[str, Dict[Device, str]]) -> None:
        """init devices and allocators."""
        devs = dev
        if not isinstance(dev, (list, tuple)):
            if not isinstance(dev, tvm.runtime.Device):
                raise TypeError(
                    "dev is expected to be Device or \
                                List[Device]"
                )
            devs = [dev]

        # CPU is required for executing shape functions
        if not any(c.device_type % RPC_SESS_MASK == tvm.cpu().device_type for c in devs):
            devs.append(tvm.cpu())

        default_alloc_type = VirtualMachine.POOLED_ALLOCATOR
        if memory_cfg is None:
            memory_cfg = {}
        elif isinstance(memory_cfg, str):
            assert memory_cfg in ["naive", "pooled"]
            if memory_cfg == "naive":
                default_alloc_type = VirtualMachine.NAIVE_ALLOCATOR
            memory_cfg = {}
        elif not isinstance(memory_cfg, dict):
            raise TypeError(
                "memory_cfg is expected be string or dictionary, "
                + "but received {}".format(type(memory_cfg))
            )
        init_args = []
        for device in devs:
            init_args.append(device.device_type % RPC_SESS_MASK)
            init_args.append(device.device_id)
            alloc_type = memory_cfg[device] if device in memory_cfg else default_alloc_type
            init_args.append(alloc_type)
        _ffi_api.VirtualMachineInit(self.module, *init_args)

    def __getitem__(self, key: str) -> PackedFunc:
        return self.module[key]


def build(mod: tvm.IRModule, target: tvm.target.Target) -> Tuple[Executable, Module]:
    """
    Build an IRModule to VM executable.

    Parameters
    ----------
    mod: IRModule
        The input IRModule to be built.

    target : tvm.target.Target
        A build target which can have optional host side compilation target.

        When TVM compiles device specific program such as CUDA,
        we also need host(CPU) side code to interact with the driver
        to setup the dimensions and parameters correctly.
        host is used to specify the host side codegen target.
        By default, llvm is used if it is enabled,
        otherwise a stackvm intepreter is used.

    Returns
    -------
    ex: tvm.relax.vm.Exectuable
        An executable that can be loaded by virtual machine.
    lib: tvm.runtime.Module
        A runtime module that contains generated code.

    Example
    -------

    .. code-block:: python
        class InputModule:
            @R.function
            def foo(x: Tensor[(3, 4), "float32"], y: Tensor[(3, 4), "float32"]):
                z = R.add(x, y)
                return z

        mod = InputModule
        target = tvm.target.Target("llvm", host="llvm")
        ex, lib = relax.vm.build(mod, target)
    """
    passes = [relax.transform.ToNonDataflow()]
    passes.append(relax.transform.CallDPSRewrite())
    passes.append(relax.transform.VMMemoryLower())
    passes.append(relax.transform.VMShapeLower())
    seq = tvm.transform.Sequential(passes)
    new_mod = seq(mod)

    # split primfunc and relax function
    rx_mod, tir_mod = _split_tir_relax(new_mod)

    lib = tvm.build(tir_mod, target)
    ex = _ffi_api.VMCodeGen(rx_mod)
    return ex, lib


def _split_tir_relax(mod: tvm.IRModule) -> Tuple[tvm.IRModule, tvm.IRModule]:
    rx_mod = IRModule({})
    tir_mod = IRModule({})
    for gv in mod.get_global_vars():
        if isinstance(mod[gv], PrimFunc):
            tir_mod[gv] = mod[gv]
        elif isinstance(mod[gv], relax.Function):
            rx_mod[gv] = mod[gv]
        else:
            raise TypeError(
                "IRModule is expected to contain PrimFunc or Function, but gets {}".format(
                    type(mod[gv])
                )
            )
    return rx_mod, tir_mod
