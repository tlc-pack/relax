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
 * \file printer/relax_script_printer.cc
 * \brief Printer class to print Relax IR to parsable Python
 */

#include <tvm/ir/type_functor.h>
#include <tvm/relax/ir_functor.h>
#include <tvm/relax/utils.h>

#include <algorithm>
#include <utility>

#include "doc.h"
#include "text_printer.h"

namespace tvm {
namespace relax {

Doc PrintDType(DataType dtype) { return Doc::StrLiteral(runtime::DLDataType2String(dtype)); }

Doc RelaxScriptPrinter::Print(const ObjectRef& node) {
  if (node->IsInstance<IRModuleNode>()) {
    return PrintIRModule(Downcast<IRModule>(node));
  } else if (node->IsInstance<TypeNode>()) {
    return VisitType(Downcast<Type>(node));
  } else if (node->IsInstance<PrimExprNode>()) {
    return VisitExpr(Downcast<PrimExpr>(node));
  } else {
    return VisitNode(node);
  }
}

Doc RelaxScriptPrinter::VisitNode_(const relay::TupleNode* op) {
  size_t num_fields = op->fields.size();

  if (num_fields == 0) {
    return Doc::Text("tuple()");
  }

  Doc doc;
  std::vector<Doc> fields;

  for (const Expr& field : op->fields) {
    fields.push_back(Print(field));
  }
  doc << "(" << Doc::Concat(fields, Doc::Text(", "));
  if (num_fields == 1) {
    doc << ",";
  }
  doc << ")";

  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relay::GlobalVarNode* op) {
  return Doc::Text(op->name_hint);
}

Doc RelaxScriptPrinter::VisitNode_(const relay::CallNode* op) {
  // TODO(@altanh): how to support when func cannot be printed as Python expr?
  //                e.g. Function or If
  Doc doc;
  std::vector<Doc> args;

  static const Op& call_tir_op = Op::Get("relax.call_tir");
  if (op->op == call_tir_op) {
    doc << "relax.call_tir";

    for (const Expr& arg : op->args) {
      args.push_back(Print(arg));
    }
    doc << "(" << Doc::Concat(args, Doc::Text(", "));

    Type output_type = op->type_args[0];
    if (const auto* out_type = output_type.as<DynTensorTypeNode>()) {
      doc << ", dtype=" << PrintDType(out_type->dtype) << ")";
    } else if (const auto* out_type = output_type.as<TupleTypeNode>()) {
      std::vector<Doc> dtypes;
      for (auto field : out_type->fields) {
        if (const auto* field_type = field.as<DynTensorTypeNode>()) {
          Doc dtype;
          dtype << PrintDType(field_type->dtype);
          dtypes.push_back(dtype);
        } else {
          LOG(FATAL) << "TypeError: Invalid type: " << field_type->GetTypeKey();
        }
      }
      doc << ", dtype=(" << Doc::Concat(dtypes, Doc::Text(", ")) << "))";
    } else {
      LOG(FATAL) << "TypeError: Invalid type: " << output_type->GetTypeKey();
    }
    return doc;
  }

  if (op->op.as<relax::ExternFuncNode>()) {
    doc << "relax.call_packed";
    args.push_back(Print(op->op));
  } else {
    doc << Print(op->op);
  }

  for (const Expr& arg : op->args) {
    args.push_back(Print(arg));
  }
  doc << "(" << Doc::Concat(args, Doc::Text(", "));

  std::vector<Doc> attrs = PrintAttrs(op->attrs);
  if (op->attrs.defined()) {
    attrs.push_back(Doc::Text("attrs_type_key=") << Doc::StrLiteral(op->attrs->GetTypeKey()));
  }
  if (!attrs.empty()) {
    doc << ", " << Doc::Concat(attrs);
  }

  if (!op->type_args.empty()) {
    doc << ", type_args=(";
    std::vector<Doc> type_args = PrintTypeArgs(op->type_args);
    doc << Doc::Concat(type_args);
    doc << ")";
  }

  doc << ")";

  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const OpNode* op) { return Doc::Text(op->name); }

Doc RelaxScriptPrinter::VisitNode_(const relay::TupleGetItemNode* op) {
  Doc doc;
  doc << Print(op->tuple) << "[" << op->index << "]";
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::VarNode* op) {
  if (!var_id_map_.count(op->vid)) {
    var_id_map_[op->vid] = GetUniqueName(op->name_hint(), "v");
  }

  return var_id_map_[op->vid];
}

/*!
 * \brief special method to print out const scalar
 * \param dtype The data type
 * \param value The value to be printed.
 */
template <typename T>
Doc ScalarLiteral(DataType dtype, const T& value) {
  std::ostringstream os;
  if (dtype == DataType::Bool()) {
    return Doc::PyBoolLiteral(value != 0);
  } else {
    os << value;
  }
  return Doc::Text(os.str());
}

// Overload of Expr printing functions
Doc RelaxScriptPrinter::PrintExpr(const Expr& expr, bool meta, bool try_inline,
                                  bool optional_info) {
  Doc printed_expr;
  if (meta) {
    printed_expr = meta_->GetMetaNode(GetRef<ObjectRef>(expr.get()));
  } else {
    printed_expr = VisitNode(expr);
  }
  return printed_expr;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::ConstantNode* op) {
  Doc doc;
  // Print out simple scalars directly.
  if (op->data->ndim == 0) {
    std::ostringstream os;
    DataType dtype = DataType(op->data->dtype);
    ICHECK_EQ(op->data->device.device_type, kDLCPU);
    auto scalar_val = ScalarLiteral(dtype, 0);
    if (dtype == DataType::Int(32)) {
      scalar_val = ScalarLiteral(dtype, static_cast<const int32_t*>(op->data->data)[0]);
    } else if (dtype == DataType::Int(64)) {
      scalar_val = ScalarLiteral(dtype, static_cast<const int64_t*>(op->data->data)[0]);
    } else if (dtype == DataType::Float(32)) {
      scalar_val = ScalarLiteral(dtype, static_cast<const float*>(op->data->data)[0]);
    } else if (dtype == DataType::Float(64)) {
      scalar_val = ScalarLiteral(dtype, static_cast<const double*>(op->data->data)[0]);
    } else if (dtype == DataType::Bool()) {
      scalar_val = ScalarLiteral(dtype, static_cast<const uint8_t*>(op->data->data)[0]);
    }
    return doc << scalar_val;
  }
  // default fall-back, record it as meta node.
  // Don't append optional_info. Because the entry function is Print,
  // and it will append the optional_info afterwards.
  return doc << PrintExpr(GetRef<Expr>(op), true, false, false);
}

Doc RelaxScriptPrinter::VisitNode_(const relax::DataflowVarNode* op) {
  if (!var_id_map_.count(op->vid)) {
    var_id_map_[op->vid] = GetUniqueName(op->name_hint(), "dv");
  }

  return var_id_map_[op->vid];
}

Doc RelaxScriptPrinter::VisitNode_(const relax::ShapeExprNode* op) {
  // TODO(@altanh): support more PrimExpr printing, and check that empty tuple
  //                is never ambiguously printed as "()"
  Doc doc;

  std::vector<Doc> fields;
  for (const PrimExpr& field : op->values) {
    fields.push_back(Print(field));
  }
  doc << "(" << Doc::Concat(fields, Doc::Text(", "));
  if (fields.size() == 1) {
    doc << ",";
  }
  doc << ")";
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::RuntimeDepShapeNode* op) {
  Doc doc;

  doc << "_";
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::MatchShapeNode* op) {
  Doc doc;
  if (op->var.defined()) {
    doc << Print(op->var) << PrintVarAnnotation(op->var) << " = ";
  }
  doc << "relax.match_shape(";
  // TODO(@altanh): maybe op->pattern should just be a ShapeExpr?
  doc << Print(op->value) << ", " << Print(relax::ShapeExpr(op->pattern));
  doc << ")";
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::VarBindingNode* op) {
  // TODO(@altanh): think deeper about normal form (need to be strict about block exprs)
  if (const relay::IfNode* ite = op->value.as<relay::IfNode>()) {
    return PrintIfStmt(op->var, GetRef<relay::If>(ite));
  } else if (const relax::FunctionNode* func = op->value.as<relax::FunctionNode>()) {
    return PrintFunctionDef(Print(op->var), GetRef<relax::Function>(func));
  } else if (const tir::PrimFuncNode* prim_func = op->value.as<tir::PrimFuncNode>()) {
    return PrintPrimFunc(op->var->name_hint(), GetRef<tir::PrimFunc>(prim_func));
  } else {
    Doc doc;
    bool print_annotation = true;
    static const Op& call_tir_op = Op::Get("relax.call_tir");
    if (const CallNode* value = op->value.as<CallNode>()) {
      if (value->op == call_tir_op) {
        print_annotation = false;
      }
    }
    doc << Print(op->var);
    if (print_annotation) {
      doc << PrintVarAnnotation(op->var);
    }
    doc << " = " << Print(op->value);
    return doc;
  }
}

Doc RelaxScriptPrinter::VisitNode_(const relax::BindingBlockNode* op) {
  Doc doc;
  for (const relax::Binding& binding : op->bindings) {
    doc << Print(binding) << Doc::NewLine();
  }
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::DataflowBlockNode* op) {
  Doc block;
  Doc body;
  std::vector<Doc> return_vars;
  for (const relax::Binding& binding : op->bindings) {
    body << Print(binding) << Doc::NewLine();
    Var var;
    if (const relax::VarBindingNode* var_binding = binding.as<relax::VarBindingNode>()) {
      var = var_binding->var;
    } else if (const relax::MatchShapeNode* shape_binding = binding.as<relax::MatchShapeNode>()) {
      var = shape_binding->var;
    }
    if (var.defined() && !var.as<relax::DataflowVarNode>()) {
      return_vars.push_back(Print(var));
    }
  }
  ICHECK(!return_vars.empty()) << "dataflow blocks should have at least one output variable";
  body << "relax.output(" << Doc::Concat(return_vars, Doc::Text(", ")) << ")";
  block << "with relax.dataflow():" << Doc::NewLine(4);
  block << Doc::Indent(4, body) << Doc::NewLine();
  return block;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::SeqExprNode* op) {
  Doc doc;
  int i = 0;
  for (const relax::BindingBlock& block : op->blocks) {
    doc << "# block " << i++ << Doc::NewLine();
    doc << Print(block);
  }
  // NOTE: the body expression is printed in the parent, since SeqExprs are used for both Function
  //       bodies and If expr bodies (which don't have a "return" statement but instead a binding)
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::FunctionNode* op) {
  Optional<String> gsymbol = op->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(gsymbol.defined());
  return PrintFunctionDef(Doc::Text(gsymbol.value()), GetRef<relax::Function>(op));
}

Doc RelaxScriptPrinter::VisitNode_(const relax::ExternFuncNode* op) {
  return Doc::StrLiteral(op->global_symbol);
}

Doc RelaxScriptPrinter::VisitExpr_(const tir::VarNode* op) {
  tir::Var var = GetRef<tir::Var>(op);
  if (!dim_var_map_.count(var)) {
    dim_var_map_[var] = GetUniqueName(var->name_hint, "dim");
  }
  return dim_var_map_[var];
}

Doc RelaxScriptPrinter::VisitExpr_(const tir::IntImmNode* op) {
  return Doc::Text(std::to_string(op->value));
}

#define TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(OpName, OpString) \
  Doc RelaxScriptPrinter::VisitExpr_(const OpName* op) {          \
    Doc doc;                                                      \
    doc << "(" << Print(op->a) << OpString;                       \
    doc << Print(op->b) << ")";                                   \
    return doc;                                                   \
  }

TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::AddNode, " + ")
TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::SubNode, " - ")
TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::MulNode, " * ")
TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::DivNode, " / ")
TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::FloorDivNode, " // ")

Doc RelaxScriptPrinter::VisitExpr_(const tir::CastNode* op) {
  Doc doc;
  doc << "tir.cast(" << PrintDType(op->dtype) << ", " << Print(op->value) << ")";
  return doc;
}

Doc RelaxScriptPrinter::VisitExpr_(const tir::MaxNode* op) {
  Doc doc;
  doc << "tir.max(" << Print(op->a) << ", " << Print(op->b) << ")";
  return doc;
}

Doc RelaxScriptPrinter::VisitType_(const relax::ShapeTypeNode* node) { return Doc::Text("Shape"); }

Doc RelaxScriptPrinter::VisitType_(const relax::ObjectTypeNode* node) {
  return Doc::Text("Object");
}

Doc RelaxScriptPrinter::VisitType_(const relax::DynTensorTypeNode* node) {
  // NOTE: to print shape information, use PrintTensorAnnotation
  return PrintTensorAnnotation(GetRef<DynTensorType>(node), NullOpt);
}

Doc RelaxScriptPrinter::VisitType_(const relay::TupleTypeNode* node) {
  if (node->fields.empty()) {
    return Doc::Text("Tuple()");
  }

  Doc doc;

  std::vector<Doc> fields;
  for (Type ty : node->fields) {
    fields.push_back(Print(ty));
  }
  doc << "Tuple(" << Doc::Concat(fields) << ")";

  return doc;
}

Doc RelaxScriptPrinter::VisitType_(const relay::FuncTypeNode* node) {
  Doc doc;
  doc << "Callable";
  if (node->type_params.size() != 0) {
    doc << "(";
    std::vector<Doc> type_params;
    for (Type type_param : node->type_params) {
      type_params.push_back(Print(type_param));
    }
    doc << Doc::Concat(type_params);
    doc << ")";
  }
  std::vector<Doc> arg_types;
  for (Type arg_type : node->arg_types) {
    arg_types.push_back(Print(arg_type));
  }
  // TODO(@yongwww): Change it to Callable[[Arg1Type, Arg2Type, ...,], ReturnType]
  //                 to be consistent with Python type hint syntax,
  return doc << "((" << Doc::Concat(arg_types) << "), " << Print(node->ret_type) << ")";
}

Doc RelaxScriptPrinter::PrintAttr(const ObjectRef& attr) {
  if (attr.defined()) {
    if (const StringObj* str = attr.as<StringObj>()) {
      return Doc::StrLiteral(GetRef<String>(str));
    } else {
      return VisitAttr(attr);
    }
  } else {
    return Doc::Text("None");
  }
}

std::vector<Doc> RelaxScriptPrinter::PrintAttrs(const Attrs& attrs) {
  std::vector<Doc> kwargs;
  if (!attrs.defined()) {
    return kwargs;
  } else if (const DictAttrsNode* dict_attrs = attrs.as<DictAttrsNode>()) {
    for (const auto& k : dict_attrs->dict) {
      kwargs.push_back(Doc::Text(k.first) << "=" << Print(k.second));
    }
  } else {
    AttrPrinter attr_printer(&kwargs, this);
    const_cast<BaseAttrsNode*>(attrs.operator->())->VisitAttrs(&attr_printer);
  }
  return kwargs;
}

std::vector<Doc> RelaxScriptPrinter::PrintTypeArgs(const Array<tvm::Type>& type_args) {
  std::vector<Doc> type_args_doc;
  if (!type_args.empty()) {
    for (const auto& type : type_args) {
      if (const auto* tensor = type.as<DynTensorTypeNode>()) {
        Doc doc;
        doc << "Tensor(ndim=" << tensor->ndim << ", dtype=" << PrintDType(tensor->dtype) << ")";
        type_args_doc.push_back(doc);
      } else {
        type_args_doc.push_back(this->VisitType(type));
      }
    }
  }
  return type_args_doc;
}

Doc RelaxScriptPrinter::VisitAttrDefault_(const Object* op) {
  return PrintAttr(GetRef<ObjectRef>(op));
}

Doc RelaxScriptPrinter::VisitAttr_(const ArrayNode* op) {
  Doc doc;
  std::vector<Doc> arr_vals;
  for (ObjectRef val : *op) {
    arr_vals.push_back(PrintAttr(val));
  }
  doc << "[" << Doc::Concat(arr_vals) << "]";
  return doc;
}

Doc RelaxScriptPrinter::VisitAttr_(const tir::IntImmNode* op) {
  return Doc::Text(std::to_string(op->value));
}

Doc RelaxScriptPrinter::VisitAttr_(const tir::FloatImmNode* op) {
  return Doc::Text(std::to_string(op->value));
}

Doc RelaxScriptPrinter::PrintIRModule(const IRModule& mod) {
  Doc doc;
  if (ShowMetaData()) {
    doc << "@tvm.script.ir_module(metadata=metadata)" << Doc::NewLine();
  } else {
    doc << "@tvm.script.ir_module" << Doc::NewLine();
  }
  doc << "class Module:";
  for (const std::pair<GlobalVar, BaseFunc>& pr : mod->functions) {
    Doc func;
    if (pr.second.as<tir::PrimFuncNode>()) {
      func = PrintPrimFunc(pr.first->name_hint, Downcast<tir::PrimFunc>(pr.second));
    } else {
      func = Print(pr.second);
    }
    doc << Doc::Indent(4, Doc::NewLine() << func);
  }
  return doc;
}

Doc RelaxScriptPrinter::PrintPrimFunc(const String& name, const tir::PrimFunc& func) {
  // we need the mod for TVMScriptPrinter to properly print the function name - maybe it's worth
  // refactoring to avoid this?
  IRModule mod;
  mod->Add(relay::GlobalVar(name), func);
  return tir::AsTVMScriptDoc(mod, "tir", false, func);
}

Doc RelaxScriptPrinter::PrintIfStmt(const relax::Var& var, const relay::If& ite) {
  const relax::SeqExprNode* true_branch = ite->true_branch.as<relax::SeqExprNode>();
  const relax::SeqExprNode* false_branch = ite->false_branch.as<relax::SeqExprNode>();
  // TODO(@altanh): this invariant must be maintained by the normal form
  ICHECK(true_branch && false_branch)
      << "in the Relax IR normal form, each branch of a If expression should be a SeqExpr";

  Doc doc;
  doc << "if " << Print(ite->cond) << ":" << Doc::NewLine(4);
  doc << Doc::Indent(4, Print(GetRef<SeqExpr>(true_branch)));
  doc << Doc::Indent(4, Print(relax::VarBinding(var, true_branch->body)));
  doc << Doc::NewLine();
  doc << "else:" << Doc::NewLine(4);
  doc << Doc::Indent(4, Print(GetRef<SeqExpr>(false_branch)));
  doc << Doc::Indent(4, Print(relax::VarBinding(var, false_branch->body)));
  return doc;
}

Doc RelaxScriptPrinter::PrintFunctionDef(const Doc& name, const relax::Function& func) {
  Doc doc;

  std::vector<Doc> params;
  for (size_t i = 0; i < func->params.size(); ++i) {
    relax::Var var = func->params[i];
    Doc param;
    param << Print(var) << PrintVarAnnotation(var);
    params.push_back(param);
  }
  if (ShowMetaData()) {
    doc << "@relax.function(metadata=metadata)" << Doc::NewLine();
  } else {
    doc << "@relax.function" << Doc::NewLine();
  }
  doc << "def " << name << "(" << Doc::Concat(params, Doc::Text(", ")) << ")";
  if (func->ret_type.defined()) {
    doc << " -> " << Print(func->ret_type);
  }
  doc << ":" << Doc::NewLine(4);

  if (const relax::SeqExprNode* body = func->body.as<relax::SeqExprNode>()) {
    doc << Doc::Indent(4, Print(func->body));
    doc << Doc::Indent(4, Doc::Text("return ") << Print(body->body)) << Doc::NewLine();
  } else if (const relax::FunctionNode* body = func->body.as<relax::FunctionNode>()) {
    // nested function
    String func_name;
    Optional<String> gsymbol = body->GetAttr<String>(tvm::attr::kGlobalSymbol);
    if (gsymbol.defined()) {
      func_name = gsymbol.value();
    } else {
      func_name = "local_func_" + std::to_string(local_func_counter_++);
    }
    Doc nested_func = PrintFunctionDef(Doc::Text(func_name), GetRef<relax::Function>(body));
    doc << Doc::Indent(4, nested_func);
    doc << Doc::Indent(4, Doc::Text("return ") << func_name) << Doc::NewLine();
  } else {
    doc << Doc::Indent(4, Doc::Text("return ") << Print(func->body)) << Doc::NewLine();
  }

  return doc;
}

Doc RelaxScriptPrinter::PrintVarAnnotation(const relax::Var& var) {
  // TODO(@altanh): we should consider moving annotation into binding
  Doc doc;
  Type annotation = var->checked_type_;
  if (annotation.defined()) {
    doc << ": ";
    if (const relax::DynTensorTypeNode* tty = annotation.as<relax::DynTensorTypeNode>()) {
      doc << PrintTensorAnnotation(GetRef<DynTensorType>(tty), var->shape_);
    } else if (const TupleTypeNode* tty = annotation.as<TupleTypeNode>()) {
      doc << PrintTupleAnnotation(GetRef<TupleType>(tty), var->shape_);
    } else {
      doc << Print(annotation);
    }
  }
  return doc;
}

Doc RelaxScriptPrinter::PrintTensorAnnotation(const relax::DynTensorType& ty,
                                              const Optional<ObjectRef>& shape) {
  Doc doc;
  doc << "Tensor(";
  // Print shape annotation
  if (shape.defined()) {
    doc << Print(Downcast<relay::Expr>(shape.value()));
  } else {
    doc << "None";
  }
  // Print dtype annotation
  doc << ", ";
  if (ty->dtype.is_void()) {
    doc << "_";
  } else {
    doc << PrintDType(ty->dtype);
  }
  // Print ndim annotation only when it cannot be inferred from shape itself.
  if (!shape.defined() || shape->IsInstance<relax::RuntimeDepShapeNode>()) {
    doc << ", ndim = " << ty->ndim;
  }
  doc << ")";
  return doc;
}

Doc RelaxScriptPrinter::PrintTupleAnnotation(const TupleType& ty,
                                             const Optional<ObjectRef>& shape) {
  Doc doc;
  doc << "Tuple";
  std::vector<Doc> fields;
  for (size_t i = 0; i < ty->fields.size(); i++) {
    if (shape) {
      if (const TupleNode* shape_tuple = shape.value().as<TupleNode>()) {
        if (const DynTensorTypeNode* type_field = ty->fields[i].as<DynTensorTypeNode>()) {
          fields.push_back(
              PrintTensorAnnotation(GetRef<DynTensorType>(type_field), shape_tuple->fields[i]));
        }
      }
    } else {
      if (const DynTensorTypeNode* type_field = ty->fields[i].as<DynTensorTypeNode>()) {
        fields.push_back(PrintTensorAnnotation(GetRef<DynTensorType>(type_field), NullOpt));
      }
    }
  }
  doc << "(" << Doc::Concat(fields, Doc::Text(", ")) << ")";
  return doc;
}

Doc RelaxScriptPrinter::GetUniqueName(std::string prefix, std::string fallback = "x") {
  if (prefix.empty()) {
    prefix = fallback;
  }
  return Doc::Text(name_table_.GetUniqueName(prefix));
}

bool RelaxScriptPrinter::ShowMetaData() { return show_meta_data_; }

String AsRelaxScript(const ObjectRef& mod, bool show_meta_data) {
  ICHECK(mod->IsInstance<relax::FunctionNode>() || mod->IsInstance<IRModuleNode>());
  Doc doc;
  runtime::TypedPackedFunc<std::string(ObjectRef)> ftyped = nullptr;
  doc << TextPrinter(show_meta_data, ftyped).PrintRelax(mod);
  return doc.str();
}

TVM_REGISTER_GLOBAL("script.AsRelaxScript").set_body_typed(AsRelaxScript);

}  // namespace relax
}  // namespace tvm
