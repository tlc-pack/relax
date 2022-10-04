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
"""The attributes node used for Relax operators"""
from tvm.ir import Attrs
import tvm._ffi


@tvm._ffi.register_object("relax.attrs.AllocTensorAttrs")
class AllocTensorAttrs(Attrs):
    """Attributes used in alloc_tensor operators"""


@tvm._ffi.register_object("relax.attrs.MemAllocStorageAttrs")
class MemAllocStorageAttrs(Attrs):
    """Attributes used in memory planning alloc_storage operators"""


@tvm._ffi.register_object("relax.attrs.MemAllocTensorAttrs")
class MemAllocTensorAttrs(Attrs):
    """Attributes used in memory planning alloc_tensor operators"""


@tvm._ffi.register_object("relax.attrs.VMAllocStorageAttrs")
class VMAllocStorageAttrs(Attrs):
    """Attributes used in VM alloc_storage operators"""


@tvm._ffi.register_object("relax.attrs.VMAllocTensorAttrs")
class VMAllocTensorAttrs(Attrs):
    """Attributes used in VM alloc_tensor operators"""


@tvm._ffi.register_object("relax.attrs.UniqueAttrs")
class UniqueAttrs(Attrs):
    """Attributes used for the unique operator"""


@tvm._ffi.register_object("relax.attrs.PrintAttrs")
class PrintAttrs(Attrs):
    """Attributes used for the print operator"""


@tvm._ffi.register_object("relax.attrs.AssertOpAttrs")
class AssertOpAttrs(Attrs):
    """Attributes used for the assert operator"""


@tvm._ffi.register_object("relax.attrs.DenseAttrs")
class DenseAttrs(Attrs):
    """Attributes for nn.dense"""


@tvm._ffi.register_object("relax.attrs.Conv2DAttrs")
class Conv2DAttrs(Attrs):
    """Attributes for nn.Conv2D"""


@tvm._ffi.register_object("relax.attrs.MaxPool2DAttrs")
class MaxPool2DAttrs(Attrs):
    """Attributes for nn.MaxPool2D"""


@tvm._ffi.register_object("relax.attrs.SoftmaxAttrs")
class SoftmaxAttrs(Attrs):
    """Attributes for nn.softmax"""
