from re import L
import yaml
from utils import LineLoader, NamespaceHelper, context
from model import (
    Location,
    FunctionSchema,
    SchemaKind,
    DispatchKey,
    SelfArgument,
    dispatch_keys,
)
from dataclasses import dataclass
from typing import Dict

# NOTE:
# Currently, there is a version mismatch between
# native_functions.yaml (v1.13.0a0) and libtorch distribution (v1.0.1).
# As the order of dps style seems to be different between two distributions,
# this parser handles it accordingly.


@dataclass(frozen=True)
class NativeFunction:
    @staticmethod
    def from_yaml(ei: Dict[str, object], loc: "Location"):
        e = ei.copy()
        funcs = e.pop("func")
        assert isinstance(funcs, str), f"not a str: {funcs}"

        namespace_helper = NamespaceHelper.from_namespaced_entity(
            namespaced_entity=funcs, max_level=1
        )
        namespace = namespace_helper.get_cpp_namespace(default="aten")
        func = FunctionSchema.parse(namespace_helper.entity_name)

        # We only care about DPS style
        # If not dps style, ignore
        if func.kind() != SchemaKind.out:
            return None, None, None, None

        ins, outs, params = [], [], None
        args = func.arguments
        for arg in args.all:
            # TODO: Assume it is input tensor. May not be true.
            if isinstance(arg, SelfArgument):
                ins.append(arg)
            else:
                annot = arg.annotation
                # TODO: there might be non-tensor arguments. Support them later.
                if (annot is not None) and (annot.is_write):
                    # output tensor
                    outs.append(arg)
                else:
                    # input tensor
                    ins.append(arg)

        # This will have (device, dps_func_name) pair
        dispatch: Dict[DispatchKey, str] = {}
        raw_dispatch = e.pop("dispatch", None)
        for ks, v in raw_dispatch.items():
            if ks == "__line__":
                continue  # not worth tracking line numbers for dispatch entries
            assert isinstance(ks, str), e
            for k in ks.split(","):
                dispatch_key = DispatchKey.parse(k.strip())
                assert dispatch_key in dispatch_keys, (
                    f"Dispatch key {dispatch_key} of kernel {v} " "is not a supported dispatch key."
                )
                # We only allow at most 2 levels of namespace for kernels.
                # We will append "native" to a custom kernel namespace.
                namespace_helper = NamespaceHelper.from_namespaced_entity(v, max_level=2)
                kernel_namespace = namespace_helper.get_cpp_namespace(default="at")
                # Why is 'structured' included? External backends (e.g.
                # XLA) opt into which ops are structured independently
                # of which in-tree ops are structured
                dispatch[dispatch_key] = (kernel_namespace, namespace_helper.entity_name)

        # TODO: For now, only support CPU impl
        dps_func_name_pair = dispatch[DispatchKey.CPU]
        return dps_func_name_pair, ins, outs, params


def parse(native_yaml_path):

    aten_ops = []
    with open(native_yaml_path, "r") as f:
        es = yaml.load(f, Loader=LineLoader)
        for e in es:
            assert isinstance(e.get("__line__"), int), e
            loc = Location(native_yaml_path, e["__line__"])
            funcs = e.get("func")

            with context(lambda: f"in {loc}:\n  {funcs}"):
                dps_func_name_pair, ins, outs, params = NativeFunction.from_yaml(e, loc)

                if dps_func_name_pair:
                    aten_ops.append((dps_func_name_pair, ins, outs, params))

    return aten_ops


def gen(aten_ops):
    with open("byo_libs.cc", "w") as ofp:
        headers = f"""#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h> 

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>     

"""
        ofp.write(headers)

        ofp.write("namespace libtorch {")
        ofp.write(
            f"""
using namespace tvm;
using namespace tvm::runtime;

"""
        )

        for (dps_func_name_pair, ins, outs, params) in aten_ops:
            namespace, func_name = dps_func_name_pair
            cpp_func_name = f"{namespace}_{func_name}"
            func_def = f"// define a wrapper function to run {namespace}::{func_name}\n"
            func_def += f"""void {cpp_func_name}("""
            args = ""
            num_ins, num_outs = len(ins), len(outs)
            # define arguments
            for i in range(num_ins):
                if len(args):
                    args += ", "
                args += f"NDArray in{i}"

            for i in range(num_outs):
                if len(args):
                    args += ", "
                args += f"NDArray out{i}"

            # define function body
            func_def += f"{args}) {{"

            # convert input tensors
            func_body = f"""    
    // convert TVM NDArray to Pytorch at::Tensor"""
            aten_op_args = ""
            for i in range(num_outs):
                func_body += f"""
    {namespace}::Tensor torch_out{i} = {namespace}::fromDLPack(out{i}.ToDLPack());"""
                if len(aten_op_args):
                    aten_op_args += ", "
                aten_op_args += f"torch_out{i}"

            for i in range(num_ins):
                func_body += f"""
    {namespace}::Tensor torch_in{i} = {namespace}::fromDLPack(in{i}.ToDLPack());"""
                if len(aten_op_args):
                    aten_op_args += ", "
                aten_op_args += f"torch_in{i}"

            func_body += f"""
    // run the torch aten operator
    {namespace}::{func_name}({aten_op_args});
"""
            func_def += f"{func_body}"
            func_def += f"""}}
// Register the function as an external function 
TVM_DLL_EXPORT_TYPED_FUNC(libtorch_{cpp_func_name}, {cpp_func_name});

"""

            ofp.write(func_def)

        ofp.write("} // namespace libtorch\n")


def parse_and_gen(native_yaml_path="native_functions.yaml"):
    gen(parse(native_yaml_path))


if __name__ == "__main__":
    parse_and_gen()
