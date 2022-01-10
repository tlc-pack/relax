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

# Example BERT inference/fine-tuning workloads by converting the forward and backward BERT Relay graph from Relay to Relax


import tvm
from tvm import relax
from tvm.relax.testing import relay_translator


if __name__ == "__main__":
    mod = relay_translator.load_text("bert_16_128.txt")

    bb = relax.BlockBuilder()
    with bb.function("main"):
        relay_translator.from_relay(mod["main"])

    mod = bb.get()
    target = tvm.target.Target("llvm", host="llvm")
    ex, lib = relax.vm.build(mod, target)
