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
 * \file src/tvm/relax/ir/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relax (inherited from Relay).
 */

#include <tvm/relax/dataflow_pattern.h>

#define RELAX_PATTERN_PRINTER_DEF(NODE_TYPE, REPR_LAMBDA)                 \
  TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)                              \
      .set_dispatch<NODE_TYPE>([](const ObjectRef& ref, ReprPrinter* p) { \
        auto* node = static_cast<const NODE_TYPE*>(ref.get());            \
        REPR_LAMBDA(p, node);                                             \
      })

namespace tvm {
namespace relax {

TVM_REGISTER_NODE_TYPE(ExternFuncPatternNode);
ExternFuncPattern::ExternFuncPattern(String global_symbol) {
  ObjectPtr<ExternFuncPatternNode> n = make_object<ExternFuncPatternNode>();
  n->global_symbol_ = std::move(global_symbol);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.ExternFuncPattern")
    .set_body_typed([](String global_symbol) { return ExternFuncPattern(global_symbol); });
RELAX_PATTERN_PRINTER_DEF(ExternFuncPatternNode, [](auto p, auto node) {
  p->stream << "ExternFuncPattern(" << node->global_symbol() << ")";
});

TVM_REGISTER_NODE_TYPE(VarPatternNode);
VarPattern::VarPattern(String name_hint) {
  ObjectPtr<VarPatternNode> n = make_object<VarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.VarPattern").set_body_typed([](String name_hint) {
  return VarPattern(name_hint);
});
RELAX_PATTERN_PRINTER_DEF(VarPatternNode, [](auto p, auto node) {
  p->stream << "VarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(DataflowVarPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DataflowVarPattern")
    .set_body_typed([](String name_hint) { return DataflowVarPattern(name_hint); });
RELAX_PATTERN_PRINTER_DEF(DataflowVarPatternNode, [](auto p, auto node) {
  p->stream << "DataflowVarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(DynTensorTypePatternNode);
DynTensorTypePattern::DynTensorTypePattern(DynTensorType type) {
  ObjectPtr<DynTensorTypePatternNode> n = make_object<DynTensorTypePatternNode>();
  n->type = std::move(type);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DynTensorTypePattern")
    .set_body_typed([](DynTensorType type) { return DynTensorTypePattern(type); });
RELAX_PATTERN_PRINTER_DEF(DynTensorTypePatternNode, [](auto p, auto node) {
  p->stream << "DynTensorTypePattern(" << node->type << ")";
});

TVM_REGISTER_NODE_TYPE(RuntimeDepShapePatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.RuntimeDepShapePattern").set_body_typed([] {
  return RuntimeDepShapePattern(make_object<RuntimeDepShapePatternNode>());
});
RELAX_PATTERN_PRINTER_DEF(RuntimeDepShapePatternNode,
                          [](auto p, auto node) { p->stream << "RuntimeDepShapePattern()"; });

TVM_REGISTER_NODE_TYPE(ExprPatternNode);
ExprPattern::ExprPattern(Expr expr) {
  ObjectPtr<ExprPatternNode> n = make_object<ExprPatternNode>();
  n->expr = std::move(expr);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.ExprPattern").set_body_typed([](Expr e) {
  return ExprPattern(e);
});
RELAX_PATTERN_PRINTER_DEF(ExprPatternNode, [](auto p, auto node) { p->Print(node->expr); });

TVM_REGISTER_NODE_TYPE(ConstantPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.ConstantPattern").set_body_typed([]() {
  auto c = ConstantPattern(make_object<ConstantPatternNode>());
  return c;
});
RELAX_PATTERN_PRINTER_DEF(ConstantPatternNode,
                          [](auto p, auto node) { p->stream << "ConstantPattern()"; });

TVM_REGISTER_NODE_TYPE(CallPatternNode);
CallPattern::CallPattern(DFPattern op, Array<DFPattern> args) {
  ObjectPtr<CallPatternNode> n = make_object<CallPatternNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.CallPattern")
    .set_body_typed([](DFPattern op, Array<DFPattern> args) { return CallPattern(op, args); });
RELAX_PATTERN_PRINTER_DEF(CallPatternNode, [](auto p, auto node) {
  p->stream << "CallPatternNode(" << node->op << ", " << node->args << ")";
});

TVM_REGISTER_NODE_TYPE(FunctionPatternNode);
FunctionPattern::FunctionPattern(Array<DFPattern> params, DFPattern body) {
  ObjectPtr<FunctionPatternNode> n = make_object<FunctionPatternNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.FunctionPattern")
    .set_body_typed([](Array<DFPattern> params, DFPattern body) {
      return FunctionPattern(params, body);
    });
RELAX_PATTERN_PRINTER_DEF(FunctionPatternNode, [](auto p, auto node) {
  p->stream << "FunctionPatternNode(" << node->params << ", " << node->body << ")";
});

TVM_REGISTER_NODE_TYPE(IfPatternNode);
IfPattern::IfPattern(DFPattern cond, DFPattern true_branch, DFPattern false_branch) {
  ObjectPtr<IfPatternNode> n = make_object<IfPatternNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.IfPattern")
    .set_body_typed([](DFPattern cond, DFPattern true_branch, DFPattern false_branch) {
      return IfPattern(cond, true_branch, false_branch);
    });
RELAX_PATTERN_PRINTER_DEF(IfPatternNode, [](auto p, auto node) {
  p->stream << "IfPattern(" << node->cond << ", " << node->true_branch << ", " << node->false_branch
            << ")";
});

TVM_REGISTER_NODE_TYPE(TuplePatternNode);
TuplePattern::TuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<TuplePatternNode> n = make_object<TuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.TuplePattern")
    .set_body_typed([](tvm::Array<DFPattern> fields) { return TuplePattern(fields); });
RELAX_PATTERN_PRINTER_DEF(TuplePatternNode, [](auto p, auto node) {
  p->stream << "TuplePattern(" << node->fields << ")";
});

TVM_REGISTER_NODE_TYPE(TupleGetItemPatternNode);
TupleGetItemPattern::TupleGetItemPattern(DFPattern tuple, int index) {
  ObjectPtr<TupleGetItemPatternNode> n = make_object<TupleGetItemPatternNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.TupleGetItemPattern")
    .set_body_typed([](DFPattern tuple, int index) { return TupleGetItemPattern(tuple, index); });
RELAX_PATTERN_PRINTER_DEF(TupleGetItemPatternNode, [](auto p, auto node) {
  p->stream << "TupleGetItemPatternNode(" << node->tuple << ", " << node->index << ")";
});

TVM_REGISTER_NODE_TYPE(AndPatternNode);
AndPattern::AndPattern(DFPattern left, DFPattern right) {
  ObjectPtr<AndPatternNode> n = make_object<AndPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.AndPattern")
    .set_body_typed([](DFPattern left, DFPattern right) { return AndPattern(left, right); });
RELAX_PATTERN_PRINTER_DEF(AndPatternNode, [](auto p, auto node) {
  p->stream << "AndPattern(" << node->left << " & " << node->right << ")";
});

TVM_REGISTER_NODE_TYPE(OrPatternNode);
OrPattern::OrPattern(DFPattern left, DFPattern right) {
  ObjectPtr<OrPatternNode> n = make_object<OrPatternNode>();
  n->left = std::move(left);
  n->right = std::move(right);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.OrPattern")
    .set_body_typed([](DFPattern left, DFPattern right) { return OrPattern(left, right); });
RELAX_PATTERN_PRINTER_DEF(OrPatternNode, [](auto p, auto node) {
  p->stream << "OrPattern(" << node->left << " | " << node->right << ")";
});

TVM_REGISTER_NODE_TYPE(WildcardPatternNode);
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.WildcardPattern").set_body_typed([]() {
  auto w = WildcardPattern(make_object<WildcardPatternNode>());
  return w;
});
RELAX_PATTERN_PRINTER_DEF(WildcardPatternNode, [](auto p, auto node) { p->stream << "*"; });

TVM_REGISTER_NODE_TYPE(TypePatternNode);
TypePattern::TypePattern(DFPattern pattern, Type type) {
  ObjectPtr<TypePatternNode> n = make_object<TypePatternNode>();
  n->pattern = std::move(pattern);
  n->type = std::move(type);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.TypePattern")
    .set_body_typed([](DFPattern pattern, Type type) { return TypePattern(pattern, type); });
RELAX_PATTERN_PRINTER_DEF(TypePatternNode, [](auto p, auto node) {
  p->stream << "TypePattern(" << node->pattern << " has type " << node->type << ")";
});

TVM_REGISTER_NODE_TYPE(ShapePatternNode);
ShapePattern::ShapePattern(DFPattern pattern, Array<PrimExpr> shape) {
  ObjectPtr<ShapePatternNode> n = make_object<ShapePatternNode>();
  n->pattern = std::move(pattern);
  n->shape = std::move(shape);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.ShapePattern")
    .set_body_typed([](DFPattern pattern, Array<PrimExpr> shape) {
      return ShapePattern(pattern, shape);
    });
RELAX_PATTERN_PRINTER_DEF(ShapePatternNode, [](auto p, auto node) {
  p->stream << "ShapePattern(" << node->pattern << " has shape " << node->shape << ")";
});

TVM_REGISTER_NODE_TYPE(DataTypePatternNode);
DataTypePattern::DataTypePattern(DFPattern pattern, DataType dtype) {
  ObjectPtr<DataTypePatternNode> n = make_object<DataTypePatternNode>();
  n->pattern = std::move(pattern);
  n->dtype = std::move(dtype);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DataTypePattern")
    .set_body_typed([](DFPattern pattern, DataType dtype) {
      return DataTypePattern(pattern, dtype);
    });
RELAX_PATTERN_PRINTER_DEF(DataTypePatternNode, [](auto p, auto node) {
  p->stream << "DataTypePattern(" << node->pattern << " has dtype " << node->dtype << ")";
});

TVM_REGISTER_NODE_TYPE(AttrPatternNode);
AttrPattern::AttrPattern(DFPattern pattern, DictAttrs attrs) {
  ObjectPtr<AttrPatternNode> n = make_object<AttrPatternNode>();
  n->pattern = std::move(pattern);
  n->attrs = std::move(attrs);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.AttrPattern")
    .set_body_typed([](DFPattern pattern, DictAttrs attrs) { return AttrPattern(pattern, attrs); });
RELAX_PATTERN_PRINTER_DEF(AttrPatternNode, [](auto p, auto node) {
  p->stream << "AttrPattern(" << node->pattern << " has attributes " << node->attrs << ")";
});

TVM_REGISTER_NODE_TYPE(DominatorPatternNode);
DominatorPattern::DominatorPattern(DFPattern parent, DFPattern path, DFPattern child) {
  ObjectPtr<DominatorPatternNode> n = make_object<DominatorPatternNode>();
  n->parent = std::move(parent);
  n->path = std::move(path);

  n->child = std::move(child);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.DominatorPattern")
    .set_body_typed([](DFPattern parent, DFPattern path, DFPattern child) {
      return DominatorPattern(parent, path, child);
    });
RELAX_PATTERN_PRINTER_DEF(DominatorPatternNode, [](auto p, auto node) {
  p->stream << "DominatorPattern(" << node->parent << ", " << node->path << ", " << node->child
            << ")";
});

// Syntatic Sugar
DFPattern DFPattern::operator()(const std::vector<DFPattern>& args) const {
  return CallPattern(GetRef<DFPattern>(this->get()), Array<DFPattern>(args));
}
DFPattern DFPattern::operator+(const DFPattern& other) const {
  return IsOp("add")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator-(const DFPattern& other) const {
  return IsOp("subtract")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator*(const DFPattern& other) const {
  return IsOp("multiply")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator/(const DFPattern& other) const {
  return IsOp("divide")({GetRef<DFPattern>(this->get()), other});
}
DFPattern DFPattern::operator||(const DFPattern& other) const {
  return OrPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::operator&&(const DFPattern& other) const {
  return AndPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::Optional(const std::function<DFPattern(const DFPattern&)>& func) const {
  DFPattern current = GetRef<DFPattern>(this->get());
  return current || func(current);
}

DFPattern DFPattern::HasAttr(const Map<String, ObjectRef>& attrs) const {
  return AttrPattern(GetRef<DFPattern>(this->get()), DictAttrs(attrs));
}
DFPattern DFPattern::HasType(const Type& type) const {
  return TypePattern(GetRef<DFPattern>(this->get()), type);
}
DFPattern DFPattern::HasDtype(const DataType& dtype) const {
  return DataTypePattern(GetRef<DFPattern>(this->get()), dtype);
}
DFPattern DFPattern::HasDtype(const std::string& dtype) const {
  return HasDtype(DataType(runtime::String2DLDataType(dtype)));
}
DFPattern DFPattern::HasShape(const Array<PrimExpr>& shape) const {
  return ShapePattern(GetRef<DFPattern>(this->get()), shape);
}
DFPattern DFPattern::IsRuntimeDepShape() const {
  return AndPattern(GetRef<DFPattern>(this->get()),
                    RuntimeDepShapePattern(make_object<RuntimeDepShapePatternNode>()));
}
DFPattern IsVar(const String& name) { return VarPattern(name); }
DFPattern IsConstant() { return ConstantPattern(make_object<ConstantPatternNode>()); }
DFPattern IsWildcard() { return WildcardPattern(make_object<WildcardPatternNode>()); }
DFPattern IsExpr(const Expr& expr) { return ExprPattern(expr); }
DFPattern IsOp(const String& op_name) { return IsExpr(Op::Get(op_name)); }
DFPattern IsTuple(const Array<DFPattern>& fields) { return TuplePattern(fields); }
DFPattern IsTupleGetItem(const DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
}

}  // namespace relax
}  // namespace tvm
