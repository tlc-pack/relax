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

class RelaxScriptPrinter : public relax::IRFunctor<Doc(const ObjectRef&)>,
                           public tir::ExprFunctor<Doc(const PrimExpr&)>,
                           public TypeFunctor<Doc(const Type&)>,
                           public AttrFunctor<Doc(const ObjectRef&)> {
 public:
  TVM_DLL Doc Print(const ObjectRef& node);

 private:
  NameTable name_table_;
  std::unordered_map<relay::Id, Doc, ObjectPtrHash, ObjectPtrEqual> var_id_map_;
  std::unordered_map<tir::Var, Doc, ObjectPtrHash, ObjectPtrEqual> dim_var_map_;

  // IR nodes inherited from Relay
  // Doc VisitNode_(const relay::ConstantNode* op) override;
  Doc VisitNode_(const relay::TupleNode* op) override;
  Doc VisitNode_(const relay::GlobalVarNode* op) override;
  Doc VisitNode_(const relay::CallNode* op) override;
  // Doc VisitNode_(const relay::IfNode* op) override;
  Doc VisitNode_(const OpNode* op) override;
  Doc VisitNode_(const relay::TupleGetItemNode* op) override;

  // IR nodes introduced by Relax
  Doc VisitNode_(const relax::VarNode* op) override;
  Doc VisitNode_(const relax::DataflowVarNode* op) override;
  Doc VisitNode_(const relax::ShapeExprNode* op) override;
  Doc VisitNode_(const relax::MatchShapeNode* op) override;
  Doc VisitNode_(const relax::VarBindingNode* op) override;
  Doc VisitNode_(const relax::BindingBlockNode* op) override;
  Doc VisitNode_(const relax::DataflowBlockNode* op) override;
  Doc VisitNode_(const relax::SeqExprNode* op) override;
  Doc VisitNode_(const relax::FunctionNode* op) override;
  Doc VisitNode_(const relax::ExternFuncNode* op) override;

  // PrimExpr nodes allowed in Relax
  Doc VisitExpr_(const tir::VarNode* op) override;
  Doc VisitExpr_(const tir::IntImmNode* op) override;
  Doc VisitExpr_(const tir::AddNode* op) override;
  Doc VisitExpr_(const tir::SubNode* op) override;
  Doc VisitExpr_(const tir::MulNode* op) override;
  Doc VisitExpr_(const tir::DivNode* op) override;
  Doc VisitExpr_(const tir::FloorDivNode* op) override;

  Doc PrintIRModule(const IRModule& mod);
  Doc PrintPrimFunc(const String& name, const tir::PrimFunc& func);

  Doc PrintIfStmt(const relax::Var& var, const relay::If& ite);
  Doc PrintFunctionDef(const Doc& name, const relax::Function& func);

  Doc PrintVarAnnotation(const relax::Var& var);
  Doc PrintTensorAnnotation(const relax::DynTensorType& ty, const Optional<ObjectRef>& shape);

  Doc VisitType_(const relax::ShapeTypeNode* node) override;
  Doc VisitType_(const relax::DynTensorTypeNode* node) override;
  Doc VisitType_(const relay::TupleTypeNode* node) override;

  Doc PrintAttr(const ObjectRef& attr);
  std::vector<Doc> PrintAttrs(const Attrs& attrs);
  Doc VisitAttrDefault_(const Object* op) override;
  Doc VisitAttr_(const ArrayNode* op) override;
  Doc VisitAttr_(const tir::IntImmNode* op) override;
  Doc VisitAttr_(const tir::FloatImmNode* op) override;

  Doc GetUniqueName(std::string prefix, std::string fallback);

  /*!
   * \brief Attribute printer which prints the attributes as kwargs in a call.
   */
  class AttrPrinter : public AttrVisitor {
   public:
    AttrPrinter(std::vector<Doc>* docs, RelaxScriptPrinter* parent) : docs(docs), parent_(parent) {}

    template <typename T>
    void PrintKV(const char* key, const T& value) {
      Doc doc;
      doc << key << "=" << value;
      docs->push_back(doc);
    }

    void Visit(const char* key, double* value) final { PrintKV(key, *value); }
    void Visit(const char* key, int64_t* value) final { PrintKV(key, *value); }
    void Visit(const char* key, uint64_t* value) final { PrintKV(key, *value); }
    void Visit(const char* key, int* value) final { PrintKV(key, *value); }
    void Visit(const char* key, bool* value) final { PrintKV(key, Doc::PyBoolLiteral(*value)); }
    void Visit(const char* key, std::string* value) final { PrintKV(key, Doc::StrLiteral(*value)); }
    void Visit(const char* key, void** value) final {
      LOG(FATAL) << "do not allow void as argument";
    }
    void Visit(const char* key, DataType* value) final {
      PrintKV(key, Doc::StrLiteral(runtime::DLDataType2String(*value)));
    }
    void Visit(const char* key, runtime::NDArray* value) final {
      LOG(FATAL) << "do not allow NDarray as argument";
    }
    void Visit(const char* key, runtime::ObjectRef* obj) final {
      PrintKV(key, parent_->PrintAttr(*obj));
    }

   private:
    std::vector<Doc>* docs;
    RelaxScriptPrinter* parent_;
  };
};

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
    doc << Print(op->var) << PrintVarAnnotation(op->var);
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
  for (const relax::BindingBlock& block : op->blocks) {
    doc << Print(block);
  }
  // NOTE: the body expression is printed in the parent, since SeqExprs are used for both Function
  //       bodies and If expr bodies (which don't have a "return" statement but instead a binding)
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::FunctionNode* op) {
  ICHECK(op->name.defined());
  return PrintFunctionDef(Doc::Text(op->name.value()->name_hint), GetRef<relax::Function>(op));
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
TVM_DEFINE_RELAX_PRINTER_PRIMEXPR_BINOP(tir::FloorDivNode, " // ");

Doc RelaxScriptPrinter::VisitType_(const relax::ShapeTypeNode* node) { return Doc::Text("Shape"); }

Doc RelaxScriptPrinter::VisitType_(const relax::DynTensorTypeNode* node) {
  // NOTE: to print shape information, use PrintTensorAnnotation
  return PrintTensorAnnotation(GetRef<DynTensorType>(node), NullOpt);
}

Doc RelaxScriptPrinter::VisitType_(const relay::TupleTypeNode* node) {
  if (node->fields.empty()) {
    return Doc::Text("Tuple[]");
  }

  Doc doc;

  std::vector<Doc> fields;
  for (Type ty : node->fields) {
    fields.push_back(Print(ty));
  }
  doc << "Tuple[" << Doc::Concat(fields) << "]";

  return doc;
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
  doc << "@tvm.script.ir_module" << Doc::NewLine();
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

  doc << "@relax.function" << Doc::NewLine();
  doc << "def " << name << "(" << Doc::Concat(params, Doc::Text(", ")) << ")";
  if (func->ret_type.defined()) {
    doc << " -> " << Print(func->ret_type);
  }
  doc << ":" << Doc::NewLine(4);

  const relax::SeqExprNode* body = func->body.as<relax::SeqExprNode>();
  ICHECK(body) << "in the Relax IR normal form, the body of a Function should be a SeqExpr";

  doc << Doc::Indent(4, Print(func->body));
  doc << Doc::Indent(4, Doc::Text("return ") << Print(body->body)) << Doc::NewLine();
  return doc;
}

Doc RelaxScriptPrinter::PrintVarAnnotation(const relax::Var& var) {
  Doc doc;
  if (var->type_annotation.defined()) {
    doc << ": ";
    if (const relax::DynTensorTypeNode* tty = var->type_annotation.as<relax::DynTensorTypeNode>()) {
      doc << PrintTensorAnnotation(GetRef<DynTensorType>(tty), var->shape_);
    } else {
      doc << Print(var->type_annotation);
    }
  }
  return doc;
}

Doc RelaxScriptPrinter::PrintTensorAnnotation(const relax::DynTensorType& ty,
                                              const Optional<ObjectRef>& shape) {
  Doc doc;
  doc << "Tensor[";
  if (shape.defined()) {
    doc << Print(Downcast<relay::Expr>(shape.value()));
  } else if (ty->rank != -1) {
    ICHECK_GE(ty->rank, 0) << "DynTensor ranks must be -1 (unknown) or nonnegative";
    std::vector<Doc> dims(ty->rank, Doc::Text("_"));
    doc << "(" << Doc::Concat(dims, Doc::Text(", "));
    if (ty->rank == 1) {
      doc << ",";
    }
    doc << ")";
  } else {
    doc << "_";
  }
  doc << ", ";
  if (ty->dtype.is_void()) {
    doc << "_";
  } else {
    doc << Doc::StrLiteral(runtime::DLDataType2String(ty->dtype));
  }
  doc << "]";
  return doc;
}

Doc RelaxScriptPrinter::GetUniqueName(std::string prefix, std::string fallback = "x") {
  if (prefix.empty()) {
    prefix = fallback;
  }
  return Doc::Text(name_table_.GetUniqueName(prefix));
}

String AsRelaxScript(const ObjectRef& mod) {
  ICHECK(mod->IsInstance<relax::FunctionNode>() || mod->IsInstance<IRModuleNode>());
  return RelaxScriptPrinter().Print(mod).str();
}

TVM_REGISTER_GLOBAL("script.AsRelaxScript").set_body_typed(AsRelaxScript);

}  // namespace relax
}  // namespace tvm
