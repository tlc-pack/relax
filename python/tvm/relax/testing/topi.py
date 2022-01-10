# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from . import _ffi_api


def mean(data, axis=None, keepdims=False, exclude=False):
    return _ffi_api.mean(data, axis, keepdims, exclude)

def variance(data, mean, axis=None, keepdims=False, exclude=False, unbiased=False):
    return _ffi_api.variance(data, mean, axis, keepdims, exclude, unbiased)

def reshape(data, newshape):
    return _ffi_api.reshape(data, newshape)

def reverse_reshape(data, newshape):
    return _ffi_api.reverse_reshape(data, newshape)

def bias_add(data, bias, axis=1):
    return _ffi_api.bias_add(data, bias, axis)

def collapse_sum(data, target_shape):
    return _ffi_api.collapse_sum(data, target_shape)