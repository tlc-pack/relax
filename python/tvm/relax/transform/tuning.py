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
"""Relax Tuning Pass API"""

from tvm._ffi import register_object
from tvm.runtime import Object
from . import _ffi_api


@register_object("relax.transform.Choice")
class Choice(Object):
    """
    A TVM object choice to support customization on the python side.
    """

    def __init__(self, f_transform, f_constr):
        """Constructor"""

        self.__init_handle_by_constructor__(
            _ffi_api.Choice, f_transform, f_constr  # type: ignore # pylint: disable=no-member
        )

    def get_transform_func(self):
        return _ffi_api.ChoiceGetTransformFunc(self)
