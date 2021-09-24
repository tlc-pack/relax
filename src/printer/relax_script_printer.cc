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

#include <algorithm>
#include <utility>

#include "doc.h"
#include "text_printer.h"

namespace tvm {
namespace relax {

class RelaxScriptPrinter : public relax::IRFunctor<Doc(const ObjectRef&)>,
                           public TypeFunctor<Doc(const Type&)> {
 public:
  TVM_DLL Doc Print(const ObjectRef& node);

 private:
  std::unordered_map<std::string, int> name_alloc_map_;
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

  Doc PrintDimVar(const tir::Var& var);
  Doc PrintIfStmt(const relax::Var& var, const relay::If& ite);
  Doc PrintFunctionDef(const Doc& name, const relax::Function& func);

  Doc PrintTensorAnnotation(const relax::DynTensorType& ty, const Optional<ObjectRef>& shape);

  Doc VisitType_(const relax::ShapeTypeNode* node) override;
  Doc VisitType_(const relax::DynTensorTypeNode* node) override;
  Doc VisitType_(const relay::TupleTypeNode* node) override;

  Doc GetUniqueName(std::string prefix, std::string fallback);
};

Doc RelaxScriptPrinter::Print(const ObjectRef& node) {
  if (node->IsInstance<TypeNode>()) {
    return VisitType(Downcast<Type>(node));
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

  for (size_t i = 0; i < num_fields; ++i) {
    fields.push_back(Print(op->fields[i]));
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
  Doc doc;

  if (const relax::ExternFuncNode* ext = op->op.as<relax::ExternFuncNode>()) {
    ICHECK_EQ(op->args.size(), 1) << "extern calls should only have one argument";
    doc << "relax.call_packed(" << Print(op->op) << ", " << Print(op->args[0]) << ")";
    return doc;
  }

  // TODO(@altanh): how to support when func cannot be printed as Python expr?
  //                e.g. Function or If
  doc << Print(op->op);
  if (op->args.empty()) {
    doc << "()";
    return doc;
  }

  std::vector<Doc> args;
  for (size_t i = 0; i < op->args.size(); ++i) {
    args.push_back(Print(op->args[i]));
  }
  doc << "(" << Doc::Concat(args, Doc::Text(", ")) << ")";

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
  for (size_t i = 0; i < op->values.size(); ++i) {
    auto val = op->values[i];
    if (const tir::VarNode* var = val.as<tir::VarNode>()) {
      fields.push_back(PrintDimVar(GetRef<tir::Var>(var)));
    } else if (const tir::IntImmNode* num = val.as<tir::IntImmNode>()) {
      fields.push_back(Doc::Text(std::to_string(num->value)));
    } else {
      LOG(FATAL) << "cannot print PrimExpr: " << val->GetTypeKey();
    }
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
  doc << "relax.match_shape(";
  // TODO(@altanh): maybe op->pattern should just be a ShapeExpr?
  doc << Print(relax::ShapeExpr(op->pattern)) << ", " << Print(op->value);
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
    // we need the mod for TVMScriptPrinter to properly print the function name - maybe it's worth
    // refactoring to avoid this?
    tir::PrimFunc prim_func_ref = GetRef<tir::PrimFunc>(prim_func);
    IRModule mod;
    mod->Add(relay::GlobalVar(op->var->name_hint()), prim_func_ref);
    return tir::AsTVMScriptDoc(mod, false, prim_func_ref);
  } else {
    Doc doc;
    doc << Print(op->var);
    if (op->var->type_annotation.defined()) {
      doc << ": ";
      if (const relax::DynTensorTypeNode* tty =
              op->var->type_annotation.as<relax::DynTensorTypeNode>()) {
        doc << PrintTensorAnnotation(GetRef<DynTensorType>(tty), op->var->shape_);
      } else {
        doc << Print(op->var->type_annotation);
      }
    }
    doc << " = " << Print(op->value);
    return doc;
  }
}

Doc RelaxScriptPrinter::VisitNode_(const relax::BindingBlockNode* op) {
  Doc doc;
  for (size_t i = 0; i < op->bindings.size(); ++i) {
    doc << Print(op->bindings[i]) << Doc::NewLine();
  }
  return doc;
}

Doc RelaxScriptPrinter::VisitNode_(const relax::DataflowBlockNode* op) {
  Doc block;
  Doc body;
  std::vector<Doc> return_vars;
  for (size_t i = 0; i < op->bindings.size(); ++i) {
    body << Print(op->bindings[i]) << Doc::NewLine();
    if (const relax::VarBindingNode* binding = op->bindings[i].as<relax::VarBindingNode>()) {
      if (!binding->var.as<relax::DataflowVarNode>()) {
        return_vars.push_back(Print(binding->var));
      }
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
  for (size_t i = 0; i < op->blocks.size(); ++i) {
    doc << Print(op->blocks[i]);
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
  for (size_t i = 0; i < node->fields.size(); ++i) {
    fields.push_back(Print(node->fields[i]));
  }
  doc << "Tuple[" << Doc::Concat(fields, Doc::Text(", ")) << "]";

  return doc;
}

Doc RelaxScriptPrinter::PrintDimVar(const tir::Var& var) {
  if (!dim_var_map_.count(var)) {
    dim_var_map_[var] = GetUniqueName(var->name_hint, "dim");
  }

  return dim_var_map_[var];
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
    param << Print(var);
    if (var->type_annotation.defined()) {
      param << ": ";
      if (const relax::DynTensorTypeNode* tty =
              var->type_annotation.as<relax::DynTensorTypeNode>()) {
        param << PrintTensorAnnotation(GetRef<DynTensorType>(tty), var->shape_);
      } else {
        param << Print(var->type_annotation);
      }
    }
    params.push_back(param);
  }

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

Doc RelaxScriptPrinter::PrintTensorAnnotation(const relax::DynTensorType& ty,
                                              const Optional<ObjectRef>& shape) {
  Doc doc;
  // doc << "Tensor["
  //     << (shape.defined() ? Print(Downcast<relay::Expr>(shape.value())) : Doc::Text("_")) << ", ";
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
  // TODO(@altanh): more robust name legalization
  std::replace(prefix.begin(), prefix.end(), '.', '_');
  std::string unique_prefix = prefix;
  auto it = name_alloc_map_.find(prefix);
  if (it != name_alloc_map_.end()) {
    while (name_alloc_map_.count(unique_prefix = prefix + std::to_string(++it->second)) > 0) {
    }
  }
  name_alloc_map_[unique_prefix] = 0;
  return Doc::Text(unique_prefix);
}

String AsRelaxScript(const ObjectRef& mod) {
  ICHECK(mod->IsInstance<relax::FunctionNode>());
  return "@tvm.script.relax\n" + RelaxScriptPrinter().Print(mod).str() + "\n";
}

TVM_REGISTER_GLOBAL("script.AsRelaxScript").set_body_typed(AsRelaxScript);

}  // namespace relax
}  // namespace tvm