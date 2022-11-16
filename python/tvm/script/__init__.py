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
"""TVM Script APIs of TVM Python Package, aimed to support TIR"""
from . import _parser, parser_v1

#############
from ._parser import ir
from ._parser import ir_module
from ._parser import parse as from_source_v2
from ._parser import tir
from ._parser import relax

#############
from .parser_v1 import from_source as from_source_v1
from .parser_v1 import ir_module as ir_module_v1
from .parser_v1 import relax as relax_v1
from .parser_v1 import tir as tir_v1

# pylint: disable=invalid-name

# ir = ir_v1
# ir_module = ir_module_v1
# tir = tir_v1
# relax = relax_v1
from_source = from_source_v2
