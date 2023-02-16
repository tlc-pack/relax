/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/cutlass/codegen.cc
 * \brief The 'custom' compilation pass for CUTLASS (invoked by the RelayToTIRTargetHook pass).
 */

#include "codegen.h"

#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <numeric>
#include <sstream>

#include "../../../transforms/compiler_function_utils.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace cutlass {

using Str2StrMap = std::unordered_map<std::string, std::string>;

static Str2StrMap dtype_map = {{"float16", "cutlass::half_t"},
                               {"float32", "float"},
                               {"int8", "int8_t"},
                               {"uint8", "uint8_t"},
                               {"int32", "int32_t"}};

constexpr const char* kAnyDim = "Any";

std::string GetDimAsStr(ObjectRef dim) {
  if (auto d = dim.as<IntImmNode>()) {
    return std::to_string(d->value);
  }
  return kAnyDim;
}

Str2StrMap Conv2dArgs(const Map<String, ObjectRef>& attrs, bool is_dgrad = false,
                      bool is_wgrad = false) {
  Str2StrMap args;
  auto arg0_dtype = std::string(attrs["arg0_dtype"].as<StringObj>()->data);
  auto arg1_dtype = std::string(attrs["arg1_dtype"].as<StringObj>()->data);
  auto ret_dtype = std::string(attrs["ret_dtype"].as<StringObj>()->data);
  args["ElementInputA"] = dtype_map.at(arg0_dtype);
  args["ElementInputB"] = dtype_map.at(arg1_dtype);
  args["ElementOutput"] = dtype_map.at(ret_dtype);
  args["op_def"] = std::string(attrs["cutlass_op_def"].as<StringObj>()->data);
  args["op_name"] = std::string(attrs["cutlass_op_name"].as<StringObj>()->data);
  args["op_type"] = std::string(attrs["op_type"].as<StringObj>()->data);

  auto arg0_shape = attrs["arg0_shape"].as<ArrayNode>();
  auto arg1_shape = attrs["arg1_shape"].as<ArrayNode>();
  auto ret_shape = attrs["ret_shape"].as<ArrayNode>();
  auto activation_shape = arg0_shape;
  auto weight_shape = arg1_shape;
  auto output_shape = ret_shape;

  if (is_dgrad) {
    activation_shape = ret_shape;
    output_shape = arg0_shape;
  } else if (is_wgrad) {
    activation_shape = arg1_shape;
    weight_shape = ret_shape;
    output_shape = arg0_shape;
  }

  args["N"] = GetDimAsStr(activation_shape->at(0));
  args["H"] = GetDimAsStr(activation_shape->at(1));
  args["W"] = GetDimAsStr(activation_shape->at(2));
  args["C"] = GetDimAsStr(activation_shape->at(3));
  args["P"] = GetDimAsStr(output_shape->at(1));
  args["Q"] = GetDimAsStr(output_shape->at(2));
  args["K"] = GetDimAsStr(output_shape->at(3));
  args["R"] = GetDimAsStr(weight_shape->at(1));
  args["S"] = GetDimAsStr(weight_shape->at(2));
  args["pad_h"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(0));
  args["pad_w"] = GetDimAsStr(attrs["padding"].as<ArrayNode>()->at(1));
  args["stride_h"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(0));
  args["stride_w"] = GetDimAsStr(attrs["strides"].as<ArrayNode>()->at(1));
  args["dilation_h"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(0));
  args["dilation_w"] = GetDimAsStr(attrs["dilation"].as<ArrayNode>()->at(1));

  return args;
}

inline void CutlassPrint(std::ostringstream& os, const std::string& stmt, int indent = 2) {
  for (int i = 0; i < indent; ++i) {
    os << " ";
  }
  os << stmt;
}

std::string Conv2dOp(const Str2StrMap& attrs, const Array<String>& func_args,
                     bool has_residual_block = false) {
  auto op_type = attrs.at("op_type");
  bool has_bias = op_type.find("bias") != std::string::npos;
  bool no_bias_scaling = op_type != "cutlass.conv2d_bias_sigmoid" &&
                         op_type != "cutlass.conv2d_bias_silu" &&
                         op_type != "cutlass.conv2d_bias_hardswish";

  const std::string op_name = attrs.at("op_name");
  std::ostringstream conv2d_decl;
  CutlassPrint(conv2d_decl, attrs.at("op_def"));
  CutlassPrint(conv2d_decl, "using Operation_" + op_name +
                                " = cutlass::conv::device::ImplicitGemmConvolution<" + op_name +
                                ">;\n");
  CutlassPrint(conv2d_decl, "using Conv2d = Operation_" + op_name + ";\n");
  CutlassPrint(conv2d_decl, "using ElementInputA = Conv2d::ElementA;\n");
  CutlassPrint(conv2d_decl, "using ElementInputB = Conv2d::ElementB;\n");
  CutlassPrint(conv2d_decl, "using ElementComputeEpilogue = Conv2d::ElementAccumulator;\n");

  auto get_dim = [&attrs](const std::string& axis, const std::string& var_name, int axis_idx) {
    if (attrs.at(axis) == kAnyDim) {
      return var_name + "->shape[" + std::to_string(axis_idx) + "]";
    } else {
      return attrs.at(axis);
    }
  };

  CutlassPrint(conv2d_decl, "int N = " + get_dim("N", func_args[0], 0) + ";\n");
  CutlassPrint(conv2d_decl, "int H = " + get_dim("H", func_args[0], 1) + ";\n");
  CutlassPrint(conv2d_decl, "int W = " + get_dim("W", func_args[0], 2) + ";\n");
  CutlassPrint(conv2d_decl, "int C = " + attrs.at("C") + ";\n");
  CutlassPrint(conv2d_decl, "int K = " + attrs.at("K") + ";\n");
  CutlassPrint(conv2d_decl, "int R = " + attrs.at("R") + ";\n");
  CutlassPrint(conv2d_decl, "int S = " + attrs.at("S") + ";\n");
  CutlassPrint(conv2d_decl, "int P = " + get_dim("P", "out0", 1) + ";\n");
  CutlassPrint(conv2d_decl, "int Q = " + get_dim("Q", "out0", 2) + ";\n");
  CutlassPrint(conv2d_decl, "int pad_h = " + attrs.at("pad_h") + ";\n");
  CutlassPrint(conv2d_decl, "int pad_w = " + attrs.at("pad_w") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_h = " + attrs.at("stride_h") + ";\n");
  CutlassPrint(conv2d_decl, "int stride_w = " + attrs.at("stride_w") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_h = " + attrs.at("dilation_h") + ";\n");
  CutlassPrint(conv2d_decl, "int dilation_w = " + attrs.at("dilation_w") + ";\n");

  const bool use_split_k = op_name.find("splitk") != std::string::npos;

  if (use_split_k) {
    std::string split_k_slices = op_name.substr(op_name.find_last_not_of("0123456789") + 1);
    CutlassPrint(conv2d_decl, "int split_k_slices = " + split_k_slices + ";\n");
  } else {
    CutlassPrint(conv2d_decl, "int split_k_slices = 1;\n");
  }

  CutlassPrint(
      conv2d_decl,
      "cutlass::conv::Conv2dProblemSize problem_size(N, H, W, C, K, R, S, P, Q, pad_h, pad_w, "
      "stride_h, stride_w, dilation_h, dilation_w, cutlass::conv::Mode::kCrossCorrelation, "
      "split_k_slices);\n");

  const std::string split_k_mode = use_split_k ? "kParallel" : "kSerial";
  CutlassPrint(conv2d_decl,
               "const cutlass::conv::SplitKMode split_k_mode = cutlass::conv::SplitKMode::" +
                   split_k_mode + ";\n");

  bool is_wgrad = op_type.find("backward_weight") != std::string::npos;
  bool is_dgrad = op_type.find("conv2d_transpose") != std::string::npos;

  ICHECK(func_args.size() >= 2);
  CutlassPrint(conv2d_decl, "void* ptr_a = (void*)(" + func_args[0] + "->data);\n");
  CutlassPrint(conv2d_decl, "void* ptr_b = (void*)(" + func_args[1] + "->data);\n");

  if (has_residual_block) {
    // TODO(masahi): This code assumes that there is always a bias_add in a residual block.
    ICHECK(func_args.size() >= 4);
    CutlassPrint(conv2d_decl, "void* ptr_bias = (void*)(" + func_args[2] + "->data);\n");
    CutlassPrint(conv2d_decl, "void* ptr_residual = (void*)(" + func_args[3] + "->data);\n");
  } else if (has_bias) {
    ICHECK(func_args.size() >= 3);
    CutlassPrint(conv2d_decl, "void* ptr_c_bias = (void*)(" + func_args[2] + "->data);\n");
  }

  CutlassPrint(conv2d_decl, "void* ptr_out = (void*)(out0->data);\n");
  CutlassPrint(conv2d_decl, "ElementComputeEpilogue alpha = ElementComputeEpilogue(1);\n");
  if ((!has_bias || no_bias_scaling) && !has_residual_block) {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(0);\n");
  } else {
    CutlassPrint(conv2d_decl, "ElementComputeEpilogue beta = ElementComputeEpilogue(1);\n");
  }
  CutlassPrint(conv2d_decl, "using cutlass::layout::TensorNHWC;\n");
  CutlassPrint(conv2d_decl,
               "auto activation_shape = TensorNHWC::packed(cutlass::make_Coord(N, H, W, C));\n");
  CutlassPrint(conv2d_decl,
               "auto weight_shape = TensorNHWC::packed(cutlass::make_Coord(K, R, S, C));\n");
  CutlassPrint(conv2d_decl,
               "auto output_oshape = TensorNHWC::packed(cutlass::make_Coord(N, P, Q, K));\n");

  if (is_wgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(weight_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(weight_shape);\n\n");
  } else if (is_dgrad) {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(output_oshape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(activation_shape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(activation_shape);\n\n");
  } else {
    CutlassPrint(conv2d_decl, "TensorNHWC layout_A(activation_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_B(weight_shape);\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_C(output_oshape);\n\n");
    CutlassPrint(conv2d_decl, "TensorNHWC layout_D(output_oshape);\n\n");
  }

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "using ElementOutput = EpilogueOutputOp::ElementOutput;\n");
  } else {
    CutlassPrint(conv2d_decl, "using ElementOutput = Conv2d::ElementC;\n");
  }

  std::string tensor_c_init = "{static_cast<ElementOutput*>(ptr_out), layout_C}";
  if (has_residual_block) {
    tensor_c_init = "{static_cast<ElementOutput*>(ptr_residual), layout_C}";
  } else if (has_bias) {
    tensor_c_init =
        "{static_cast<ElementOutput*>(ptr_c_bias), cutlass::layout::TensorNHWC::Stride(0)}";
  }

  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> tensor_c" + tensor_c_init + ";\n");
  CutlassPrint(conv2d_decl,
               "cutlass::TensorRef<ElementOutput, TensorNHWC> "
               "tensor_d{static_cast<ElementOutput*>(ptr_out),layout_D};\n");

  CutlassPrint(conv2d_decl, "typename Conv2d::Arguments arguments{\n");
  CutlassPrint(conv2d_decl, " problem_size,\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputA*>(ptr_a), layout_A},\n");
  CutlassPrint(conv2d_decl, " {static_cast<ElementInputB*>(ptr_b), layout_B},\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
    CutlassPrint(conv2d_decl, "{nullptr, TensorNHWC()},\n");
  } else {
    CutlassPrint(conv2d_decl, " tensor_c,\n");
    CutlassPrint(conv2d_decl, " tensor_d,\n");
  }

  if (has_residual_block) {
    ICHECK(use_split_k == false) << "Split-k not supported for residual block fusion";
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "cutlass::conv::SplitKMode::kSerial,\n");  // split_k_slices
    CutlassPrint(conv2d_decl, "static_cast<ElementOutput*>(ptr_bias),\n");
    CutlassPrint(conv2d_decl, "nullptr, 0, K};\n");
  } else if (has_bias && no_bias_scaling) {
    CutlassPrint(conv2d_decl, " {alpha},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  } else {
    CutlassPrint(conv2d_decl, "{alpha, beta},\n");
    CutlassPrint(conv2d_decl, "split_k_mode\n};\n");
  }

  CutlassPrint(conv2d_decl, "Conv2d conv2d_op;\n");

  CutlassPrint(conv2d_decl, "size_t workspace_size = conv2d_op.get_workspace_size(arguments);\n");
  // Allocate workspace memory
  CutlassPrint(conv2d_decl,
               "cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);\n");
  // Check the problem size is supported or not
  CutlassPrint(conv2d_decl, "cutlass::Status status = conv2d_op.can_implement(arguments);\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl,
                 "arguments.ref_D.reset(reinterpret_cast<ElementComputeEpilogue*>(workspace.get()),"
                 " layout_D);\n\n");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CutlassPrint(conv2d_decl, "status = conv2d_op.initialize(arguments, workspace.get());\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(
        conv2d_decl,
        "arguments.output_op = {ElementComputeEpilogue(1), ElementComputeEpilogue(0)}; \n");
    CutlassPrint(conv2d_decl, "status = conv2d_op.update(arguments, workspace.get()); \n");
    CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");
  }

  // Launch initialized CUTLASS kernel
  CutlassPrint(conv2d_decl, "status = conv2d_op();\n");
  CutlassPrint(conv2d_decl, "CHECK(status == cutlass::Status::kSuccess);\n\n");

  if (use_split_k) {
    CutlassPrint(conv2d_decl, "ReductionDevice reduction_op;\n");
    CutlassPrint(conv2d_decl,
                 "const static cutlass::conv::Operator kConvolutionalOperator = "
                 "Conv2d::kConvolutionalOperator;\n");
    CutlassPrint(conv2d_decl, "typename ReductionDevice::Arguments reduction_args(\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, "
                 "problem_size).mn(),\n");
    CutlassPrint(conv2d_decl, "problem_size.split_k_slices,\n");
    CutlassPrint(conv2d_decl,
                 "cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, "
                 "problem_size),\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl,
                 " reinterpret_cast<Conv2d::ElementAccumulator*> (workspace.get()),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::UnderlyingKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_d.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_d.stride()[Conv2d::UnderlyingKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "{\n");
    CutlassPrint(conv2d_decl, "tensor_c.data(),\n");
    CutlassPrint(conv2d_decl,
                 "ReductionStrideIndex(tensor_c.stride()[Conv2d::UnderlyingKernel::"
                 "kTensorCStrideIdx])\n");
    CutlassPrint(conv2d_decl, "},\n");
    CutlassPrint(conv2d_decl, "   {alpha, beta}\n");
    CutlassPrint(conv2d_decl, ");\n\n");
    CutlassPrint(conv2d_decl, "status = reduction_op.initialize(reduction_args, nullptr);\n");
    CutlassPrint(conv2d_decl, "status = reduction_op();\n");
  }

  return conv2d_decl.str();
}

std::string EmitSignature(const std::vector<Output>& out, const std::string& func_id,
                          const std::vector<std::string>& arg_names) {
  std::ostringstream code_stream_;
  code_stream_ << "void " << func_id << "_(";
  for (const auto& arg_name : arg_names) {
    code_stream_ << "DLTensor* " << arg_name << ", ";
  }
  for (size_t i = 0; i < out.size() - 1; ++i) {
    code_stream_ << "DLTensor* out" << i << ", ";
  }
  code_stream_ << "DLTensor* out" << out.size() - 1 << ")";
  return code_stream_.str();
}

runtime::Module Finalize(const std::string& code, const Array<String>& func_names) {
  ICHECK(!func_names.empty())
      << "Should only create CUTLASS CSourceModule if have at least one CUTLASS partition";

  std::ostringstream default_headers;
  default_headers << "#include <tvm/runtime/packed_func.h>\n";
  default_headers << "#include <dlpack/dlpack.h>\n";
  default_headers << "#include <cuda_fp16.h>\n";
  default_headers << "#include <cutlass/cutlass.h>\n";
  default_headers << "#include <cutlass/coord.h>\n";
  default_headers << "#include <cutlass/tensor_ref.h>\n";
  default_headers << "#include <cutlass/util/host_tensor.h>\n";

  const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
  ICHECK(pf != nullptr) << "Cannot find CSource module to create the external runtime module";
  VLOG(1) << "Generated CUTLASS code:" << std::endl << code;
  return (*pf)(default_headers.str() + code, "cu", func_names, /*const_vars=*/Array<String>());
}

bool IsConv2dResidualBlock(const std::string& func_name) {
  return func_name.find("conv2d") != std::string::npos &&
         func_name.find("residual") != std::string::npos;
}

GenerateBodyOutput GenerateBody(const std::string& func_name, const std::string& ext_func_id,
                                const std::vector<std::string>& output_types,
                                const Array<String>& func_args, const Map<String, ObjectRef>& attrs,
                                int* buf_idx) {
  // Make function call with input buffers when visiting arguements
  ICHECK_GT(func_args.size(), 0);
  std::ostringstream decl_stream;
  decl_stream << "(" << func_args[0];
  for (size_t i = 1; i < func_args.size(); ++i) {
    decl_stream << ", " << func_args[i];
  }
  GenerateBodyOutput ret;
  for (const auto& out_type : output_types) {
    const std::string out = "out" + std::to_string(*buf_idx++);
    decl_stream << ", " << out;
    Output output;
    output.name = out;
    output.dtype = out_type;
    output.need_copy = false;
    ret.outputs.push_back(output);
  }
  decl_stream << ");";

  const auto* instantiate_template_func =
      runtime::Registry::Get("contrib.cutlass.instantiate_template");
  ICHECK(instantiate_template_func);

  if (func_name.find("dense") != std::string::npos ||
      func_name.find("matmul") != std::string::npos) {
    Array<String> code_and_headers = (*instantiate_template_func)(func_name, attrs, func_args);
    ret.decl = code_and_headers[0];
    ret.headers = Array<String>(code_and_headers.begin() + 1, code_and_headers.end());
  } else if (func_name.find("conv2d_transpose") != std::string::npos) {
    ret.decl = Conv2dOp(Conv2dArgs(attrs, true, false), func_args, true);
  } else if (func_name.find("backward_weight") != std::string::npos) {
    ret.decl = Conv2dOp(Conv2dArgs(attrs, false, true), func_args, true);
  } else if (IsConv2dResidualBlock(func_name)) {
    ret.decl = Conv2dOp(Conv2dArgs(attrs), func_args, true);
  } else if (func_name.find("conv2d") != std::string::npos) {
    ret.decl = Conv2dOp(Conv2dArgs(attrs), func_args);
  }

  return ret;
}

namespace {

/*! \brief Return the "cutlass" Target instance to use to guide compilation. */
Target GetCutlassTarget() {
  Target target = Target::Current(/*allow_not_defined=*/true);
  if (!target.defined() || target->kind->name != "cutlass") {
    // Use the default CUTLASS compilation options if no specific "cutlass" target was given
    // in the overall targets list. In that case target_hooks.cc will invoke the custom pass
    // without pushing any target instance onto the implicit target stack.
    target = Target("cutlass");
  }
  return target;
}

class CodegenCutlass : public backend::MemoizedExprTranslator<std::vector<Output>>,
                       public CodegenCBase {
 public:
  CodegenCutlass(const std::string& id, const Map<String, ObjectRef>& attrs) {
    this->ext_func_id_ = id;
    this->attrs_ = attrs;
  }

  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "Cutlass codegen doesn't support: " << op->GetTypeKey();
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    const auto* func = call->op.as<FunctionNode>();
    ICHECK(func) << "Only composite function is supported for CUTLASS.";
    GenerateBodyOutput ret = GenerateCompositeFunctionCall(func, call);
    ext_func_body_.push_back(ret.decl);
    headers_ = ret.headers;
    return ret.outputs;
  }

  std::string JIT(const std::vector<Output>& out) {
    std::vector<std::string> arg_names;
    for (const auto& arg : ext_func_args_) {
      arg_names.push_back(arg->name_hint());
    }

    code_stream_ << EmitSignature(out, ext_func_id_, arg_names) << "{\n";

    this->EnterScope();

    // Function body
    for (auto decl : buf_decl_) {
      this->PrintIndents();
      code_stream_ << decl << "\n";
    }
    code_stream_ << "\n";
    for (auto stmt : ext_func_body_) {
      this->PrintIndents();
      code_stream_ << stmt << "\n";
    }

    this->ExitScope();
    code_stream_ << "}\n";

    this->GenerateBackendCFunc(ext_func_id_, ext_func_args_, /*const_arr_name=*/"", out, true);
    return code_stream_.str();
  }

  Array<String> GetHeaders() { return headers_; }

 private:
  Array<String> GetArgumentNames(const CallNode* call) {
    Array<String> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  // Is node `x` an ancestor of `y`?
  bool IsAncestor(const CallNode* x, const CallNode* y) {
    if (x == y) return true;
    for (auto arg : y->args) {
      const CallNode* arg_ptr = arg.as<CallNode>();
      if (arg_ptr && IsAncestor(x, arg_ptr)) return true;
    }
    return false;
  }

  GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                   const CallNode* caller) {
    const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
    ICHECK(pattern_name.defined()) << "Only functions with composite attribute are supported.";

    if (IsConv2dResidualBlock(pattern_name.value())) {
      const CallNode* current_call = callee->body.as<CallNode>();
      bool has_relu = current_call->args.size() == 1;
      const CallNode* binop = has_relu ? current_call->args[0].as<CallNode>() : current_call;
      ICHECK(binop->args.size() == 2);
      // Figure out which of the first or second argument corresponds to the residual input
      // The root conv2d call can be reached via the other input of the binary op
      int residual_index;
      if (binop->args[1].as<VarNode>()) {
        residual_index = 1;
      } else if (binop->args[0].as<VarNode>()) {
        residual_index = 0;
      } else {
        const CallNode* lhs = binop->args[0].as<CallNode>();
        const CallNode* rhs = binop->args[1].as<CallNode>();
        ICHECK(lhs && rhs);
        // The residual input should be an ancestor of the non-residual input
        residual_index = IsAncestor(rhs, lhs) ? 1 : 0;
      }
      const auto residual_input = binop->args[residual_index];
      auto call_args = GetArgumentNames(caller);
      auto func_args = call_args;
      if (call_args.size() == 3) {
        // TODO(masahi): This code assumes that there is always a bias_add in a residual block.
        for (size_t i = 0; i < call_args.size(); ++i) {
          if (callee->params[i] == residual_input) {
            auto residual_input_name = call_args[i];
            func_args.push_back(residual_input_name);
          }
        }
      } else {
        ICHECK_EQ(func_args.size(), 4) << "Residual block fusion expects 4 input tensors: data, "
                                          "weight, bias, and residual tensor.";
      }
      return GenerateBody(caller, pattern_name.value(), func_args, attrs_);
    } else {
      return GenerateBody(caller, pattern_name.value(), attrs_);
    }

    LOG(FATAL) << "Unknown composite function: " << pattern_name;
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const Array<String>& func_args,
                                  const Map<String, ObjectRef>& attrs) {
    std::vector<Type> out_types;
    if (call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        ICHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (call->checked_type()->IsInstance<TensorTypeNode>()) {
      ICHECK(call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(call->checked_type(), false);
    }

    std::vector<std::string> out_types_str;
    for (const auto& out_type : out_types) {
      out_types_str.push_back(GetDtypeString(out_type.as<TensorTypeNode>()));
    }

    return cutlass::GenerateBody(func_name, ext_func_id_, out_types_str, func_args, attrs,
                                 &buf_idx_);
  }

  GenerateBodyOutput GenerateBody(const CallNode* call, const std::string& func_name,
                                  const Map<String, ObjectRef>& attrs) {
    auto func_args = GetArgumentNames(call);
    return GenerateBody(call, func_name, func_args, attrs);
  }

  /*! \brief The id of the external cutlass ext_func. */
  std::string ext_func_id_;
  /*! \brief The attrs of the external cutlass ext_func. */
  Map<String, ObjectRef> attrs_;
  /*!
   * \brief The index to track the output buffer. Each kernel will redirect the
   * output to a buffer that may be consumed by other kernels.
   */
  int buf_idx_{0};
  /*! \brief The arguments used by a wrapped function that calls CUTLASS kernels. */
  Array<Var> ext_func_args_;
  /*! \brief Statement of the function that will be compiled using CUTLASS kernels. */
  std::vector<std::string> ext_func_body_;
  /*! \brief The declaration of intermediate buffers. */
  std::vector<std::string> buf_decl_;
  Array<String> headers_;
};  // class CodegenCutlass

class CutlassModuleCodegen {
 public:
  explicit CutlassModuleCodegen(IRModule mod) : mod_(std::move(mod)) {}

  runtime::Module CreateCSourceModule() {
    for (const auto& entry : mod_->functions) {
      if (const auto* function_node = GetCutlassFunctionNode(entry.second)) {
        GenCutlassFunc(GetRef<Function>(function_node));
      }
    }
    return Finalize(code_stream_.str(), func_names_);
  }

 private:
  void GenCutlassFunc(const Function& function) {
    ICHECK(function.defined()) << "Input error: expect a Relay function.";

    // Record the external symbol for runtime lookup.
    Optional<String> opt_global_symbol = function->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(opt_global_symbol.defined())
        << "CUTLASS functions must have a " << tvm::attr::kGlobalSymbol << " attribute";
    std::string sid = opt_global_symbol.value();
    if (std::find(func_names_.begin(), func_names_.end(), sid) != func_names_.end()) {
      // Already emitted.
      return;
    }
    func_names_.push_back(sid);

    const auto* attrs = function->attrs.as<DictAttrsNode>();
    ICHECK(attrs != nullptr);
    const auto dict = attrs->dict;
    CodegenCutlass builder(sid, dict);
    VLOG(1) << "Creating cutlass C code for '" << sid << "' from:\n" << PrettyPrint(function);
    auto out = builder.VisitExpr(function->body);
    auto code = builder.JIT(out);
    for (const auto& header : builder.GetHeaders()) {
      code_stream_ << "#include <" << header << ">\n";
    }
    code_stream_ << "\n" + code;
  }

  /*!
   * \brief Returns \p expr as function if it is a \p Function with "Compiler" attribute
   * value "cutlass".
   */
  static const FunctionNode* GetCutlassFunctionNode(const Expr& expr) {
    if (const auto* function_node = expr.as<FunctionNode>()) {
      Optional<String> opt_compiler = function_node->GetAttr<String>(attr::kCompiler);
      if (opt_compiler.defined() && opt_compiler.value() == "cutlass") {
        return function_node;
      }
    }
    return nullptr;
  }

  /*! \brief Module we are compiling. */
  IRModule mod_;
  /*! \brief The accumulated code stream that will be compiled by NVCC */
  std::ostringstream code_stream_;
  /*! \brief The accumulated function names. */
  Array<String> func_names_;
};  // CutlassModuleCodegen

/*!
 * \brief A small shim to redirect to the 'relay.ext.cutlass.compile_for_cutlass' Python
 * function which does the main CUTLASS training, c-code generation and compilation steps.
 */
tvm::transform::Pass CompileForCutlassImpl() {
  auto pass_func = [=](IRModule mod, const tvm::transform::PassContext& pass_ctx) {
    VLOG(1) << "CompileForCutlass input:" << std::endl << PrettyPrint(mod);
    const auto* pf = runtime::Registry::Get("relay.ext.cutlass.compile_for_cutlass");
    ICHECK(pf != nullptr) << "Cannot find compile_for_cutlass function";
    Target target = GetCutlassTarget();
    runtime::Module runtime_mod = (*pf)(mod, target);
    Array<runtime::Module> external_mods =
        mod->GetAttr<Array<runtime::Module>>(tvm::attr::kExternalMods).value_or({});
    external_mods.push_back(runtime_mod);
    return WithAttr(mod, tvm::attr::kExternalMods, external_mods);
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "CompileForCutlass", {});
}

runtime::Module CreateCSourceModule(const IRModule& mod) {
  VLOG(1) << "Creating CUTLASS CSource module from:" << std::endl << PrettyPrint(mod);
  return CutlassModuleCodegen(mod).CreateCSourceModule();
}

}  // namespace

TVM_REGISTER_GLOBAL("relay.ext.cutlass.create_c_source_module").set_body_typed(CreateCSourceModule);

tvm::transform::Pass CompileForCutlass() {
  return transform::Sequential(
      {transform::OutlineCompilerFunctionsWithExistingGlobalSymbols("cutlass"),
       CompileForCutlassImpl(), transform::MarkCompilerFunctionsAsExtern("cutlass")});
}

}  // namespace cutlass
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
