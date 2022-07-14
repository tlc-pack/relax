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
 * \file src/relax/ir/dataflow_pattern.cc
 * \brief The dataflow pattern language for Relax (inherited from Relay).
 */

#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/dataflow_pattern_functor.h>

#include <memory>

#include "tvm/runtime/memory.h"

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
DataflowVarPattern::DataflowVarPattern(String name_hint) {
  ObjectPtr<DataflowVarPatternNode> n = make_object<DataflowVarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
RELAX_PATTERN_PRINTER_DEF(DataflowVarPatternNode, [](auto p, auto node) {
  p->stream << "DataflowVarPattern(" << node->name_hint() << ")";
});

TVM_REGISTER_NODE_TYPE(GlobalVarPatternNode);
GlobalVarPattern::GlobalVarPattern(String name_hint) {
  ObjectPtr<GlobalVarPatternNode> n = make_object<GlobalVarPatternNode>();
  n->name = std::move(name_hint);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.GlobalVarPattern").set_body_typed([](String name_hint) {
  return GlobalVarPattern(name_hint);
});
RELAX_PATTERN_PRINTER_DEF(GlobalVarPatternNode, [](auto p, auto node) {
  p->stream << "GlobalVarPattern(" << node->name_hint() << ")";
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

TVM_REGISTER_NODE_TYPE(PrimArrPatternNode);
PrimArrPattern::PrimArrPattern(Array<PrimExpr> arr) {
  ObjectPtr<PrimArrPatternNode> n = make_object<PrimArrPatternNode>();
  n->array = std::move(arr);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.PrimArrPattern")
    .set_body_typed([](Array<PrimExpr> arr) { return PrimArrPattern(std::move(arr)); });
RELAX_PATTERN_PRINTER_DEF(PrimArrPatternNode, [](auto p, auto node) {
  p->stream << "PrimArrPattern(" << node->array << ")";
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

TVM_REGISTER_NODE_TYPE(UnorderedTuplePatternNode);
UnorderedTuplePattern::UnorderedTuplePattern(tvm::Array<DFPattern> fields) {
  ObjectPtr<UnorderedTuplePatternNode> n = make_object<UnorderedTuplePatternNode>();
  n->fields = std::move(fields);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.UnorderedTuplePattern")
    .set_body_typed([](tvm::Array<DFPattern> fields) { return UnorderedTuplePattern(fields); });
RELAX_PATTERN_PRINTER_DEF(UnorderedTuplePatternNode, [](auto p, auto node) {
  p->stream << "UnorderedTuplePattern(" << node->fields << ")";
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

TVM_REGISTER_NODE_TYPE(NotPatternNode);
NotPattern::NotPattern(DFPattern reject) {
  ObjectPtr<NotPatternNode> n = make_object<NotPatternNode>();
  n->reject = std::move(reject);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.NotPattern").set_body_typed([](DFPattern reject) {
  return NotPattern(reject);
});
RELAX_PATTERN_PRINTER_DEF(NotPatternNode,
                          [](auto p, auto node) { p->stream << "!(" << node->reject << ")"; });

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

class DFPatternDuplicator : public DFPatternFunctor<DFPattern(const DFPattern&)> {
 public:
  DFPattern VisitDFPattern(const DFPattern& pattern) override {
    return DFPatternFunctor::VisitDFPattern(pattern);
  }
  DFPattern VisitDFPattern_(const OrPatternNode* op) override {
    return OrPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const AndPatternNode* op) override {
    return AndPattern(op->left, op->right);
  }
  DFPattern VisitDFPattern_(const NotPatternNode* op) override { return NotPattern(op->reject); }
  DFPattern VisitDFPattern_(const VarPatternNode* op) override { return VarPattern(op->name); }
  DFPattern VisitDFPattern_(const ConstantPatternNode* op) override {
    return ConstantPattern(make_object<ConstantPatternNode>());
  }
  DFPattern VisitDFPattern_(const WildcardPatternNode* op) override {
    return WildcardPattern(make_object<WildcardPatternNode>());
  }
  DFPattern VisitDFPattern_(const ExprPatternNode* op) override { return ExprPattern(op->expr); }
  DFPattern VisitDFPattern_(const GlobalVarPatternNode* op) override {
    return GlobalVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const TuplePatternNode* op) override {
    return TuplePattern(op->fields);
  }
  DFPattern VisitDFPattern_(const UnorderedTuplePatternNode* op) override {
    return UnorderedTuplePattern(op->fields);
  }
  DFPattern VisitDFPattern_(const TupleGetItemPatternNode* op) override {
    return TupleGetItemPattern(op->tuple, op->index);
  }
  DFPattern VisitDFPattern_(const CallPatternNode* op) override {
    return CallPattern(op->op, op->args);
  }
  DFPattern VisitDFPattern_(const DataTypePatternNode* op) override {
    return DataTypePattern(op->pattern, op->dtype);
  }
  DFPattern VisitDFPattern_(const FunctionPatternNode* op) override {
    return FunctionPattern(op->params, op->body);
  }
  DFPattern VisitDFPattern_(const ShapePatternNode* op) override {
    return ShapePattern(op->pattern, op->shape);
  }
  DFPattern VisitDFPattern_(const TypePatternNode* op) override {
    return TypePattern(op->pattern, op->type);
  }
  DFPattern VisitDFPattern_(const RuntimeDepShapePatternNode* op) override {
    return RuntimeDepShapePattern(make_object<RuntimeDepShapePatternNode>());
  }
  DFPattern VisitDFPattern_(const DataflowVarPatternNode* op) override {
    return DataflowVarPattern(op->name);
  }
  DFPattern VisitDFPattern_(const ExternFuncPatternNode* op) override {
    return ExternFuncPattern(op->global_symbol());
  }
  DFPattern VisitDFPattern_(const PrimArrPatternNode* op) override {
    return PrimArrPattern(op->array);
  }
};

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
DFPattern DFPattern::operator|(const DFPattern& other) const {
  return OrPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::operator&(const DFPattern& other) const {
  return AndPattern(GetRef<DFPattern>(this->get()), other);
}

DFPattern DFPattern::operator~() const { return NotPattern(GetRef<DFPattern>(this->get())); }

DFPattern DFPattern::Optional(const std::function<DFPattern(const DFPattern&)>& func) const {
  DFPattern current = GetRef<DFPattern>(this->get());
  return current | func(current);
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
DFPattern DFPattern::HasRuntimeDepShape() const {
  return GetRef<DFPattern>(this->get()) &
         RuntimeDepShapePattern(make_object<RuntimeDepShapePatternNode>());
}

DFPattern::operator UsedBySeq() const { return UsedBySeq{{*this}}; }
DFPattern::operator OnlyUsedBySeq() const { return OnlyUsedBySeq{{*this}}; }

std::stack<PatternContext>& pattern_ctx_stack() {
  thread_local std::stack<PatternContext> graph_pattern_managers;
  return graph_pattern_managers;
}

PatternContext PatternContext::Current() {
  ICHECK(!pattern_ctx_stack().empty()) << "No active PatternContext found.";
  return pattern_ctx_stack().top();
}

void PatternContext::EnterWithScope() { pattern_ctx_stack().push(*this); }

void PatternContext::ExitWithScope() {
  ICHECK(pattern_ctx_stack().top().same_as(*this));
  pattern_ctx_stack().pop();
}

static void sync_graph_constraints(const DFPattern& lhs, const DFPattern& rhs, PairCons pcon) {
  PatternContext::Current().add_constraint(lhs, rhs, pcon);
}

TVM_REGISTER_NODE_TYPE(UsedBySeqNode);
UsedBySeq::UsedBySeq(Array<DFPattern> patterns) {
  for (size_t i = 1; i < patterns.size(); ++i)
    sync_graph_constraints(patterns[i - 1], patterns[i], PairCons{PairCons::kUsedBy, -1});

  ObjectPtr<UsedBySeqNode> n = make_object<UsedBySeqNode>();
  n->patterns = std::move(patterns);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.UsedBySeq")
    .set_body_typed([](Array<DFPattern> patterns) { return UsedBySeq(std::move(patterns)); });
RELAX_PATTERN_PRINTER_DEF(UsedBySeqNode, [](auto p, auto node) {
  p->stream << "(";
  for (size_t i = 0; i < node->patterns.size(); ++i) {
    if (i != 0) p->stream << " ^ ";
    p->stream << node->patterns[i];
  }
  p->stream << ")";
});

TVM_REGISTER_NODE_TYPE(OnlyUsedBySeqNode);
OnlyUsedBySeq::OnlyUsedBySeq(Array<DFPattern> patterns) {
  for (size_t i = 1; i < patterns.size(); ++i)
    sync_graph_constraints(patterns[i - 1], patterns[i], PairCons{PairCons::kOnlyUsedBy, -1});

  ObjectPtr<OnlyUsedBySeqNode> n = make_object<OnlyUsedBySeqNode>();
  n->patterns = std::move(patterns);
  data_ = std::move(n);
}
TVM_REGISTER_GLOBAL("relax.dataflow_pattern.OnlyUsedBySeq")
    .set_body_typed([](Array<DFPattern> patterns) { return OnlyUsedBySeq(std::move(patterns)); });
RELAX_PATTERN_PRINTER_DEF(OnlyUsedBySeqNode, [](auto p, auto node) {
  p->stream << "(";
  for (size_t i = 0; i < node->patterns.size(); ++i) {
    if (i != 0) p->stream << " >> ";
    p->stream << node->patterns[i];
  }
  p->stream << ")";
});

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.used_by")
    .set_body_typed([](UsedBySeq lhs, UsedBySeq rhs, int index) { return lhs.UsedBy(rhs, index); });

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.only_used_by")
    .set_body_typed([](OnlyUsedBySeq lhs, OnlyUsedBySeq rhs, int index) {
      return lhs.OnlyUsedBy(rhs, index);
    });

UsedBySeq UsedBy(const UsedBySeq& lhs, const UsedBySeq& rhs, int index) {
  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(),
                         PairCons{PairCons::kUsedBy, index});

  Array<DFPattern> ret;
  ret.reserve(lhs->patterns.size() + rhs->patterns.size());
  ret.insert(ret.end(), lhs->patterns.begin(), lhs->patterns.end());
  ret.insert(ret.end(), rhs->patterns.begin(), rhs->patterns.end());
  return UsedBySeq(std::move(ret));
}
UsedBySeq operator^(const UsedBySeq& lhs, const UsedBySeq& rhs) { return lhs.UsedBy(rhs); }

OnlyUsedBySeq OnlyUsedBy(const OnlyUsedBySeq& lhs, const OnlyUsedBySeq& rhs, int index) {
  sync_graph_constraints(lhs->patterns.back(), rhs->patterns.front(),
                         PairCons{PairCons::kOnlyUsedBy, index});
  Array<DFPattern> ret;
  ret.reserve(lhs->patterns.size() + rhs->patterns.size());
  ret.insert(ret.end(), lhs->patterns.begin(), lhs->patterns.end());
  ret.insert(ret.end(), rhs->patterns.begin(), rhs->patterns.end());
  return OnlyUsedBySeq(std::move(ret));
}
OnlyUsedBySeq operator>>(const OnlyUsedBySeq& lhs, const OnlyUsedBySeq& rhs) {
  return lhs.OnlyUsedBy(rhs);
}

DFPattern IsVar(const String& name) { return VarPattern(name); }
DFPattern IsConstant() { return ConstantPattern(make_object<ConstantPatternNode>()); }
DFPattern IsWildcard() { return WildcardPattern(make_object<WildcardPatternNode>()); }
DFPattern IsExpr(const Expr& expr) { return ExprPattern(expr); }
DFPattern IsOp(const String& op_name) { return IsExpr(Op::Get(op_name)); }
DFPattern IsCallTIR(const String& name, Optional<TuplePattern> var_args,
                    Optional<Array<PrimExpr>> oshape) {
  DFPattern arg_pattern;
  if (!var_args.defined()) {
    arg_pattern = IsWildcard();
  } else {
    arg_pattern = var_args.value();
  }

  DFPattern shape_pattern;
  if (!oshape.defined()) {
    shape_pattern = IsWildcard();
  } else {
    shape_pattern = PrimArrPattern(oshape.value());
  }

  return IsOp("relax.call_tir")(GlobalVarPattern(name), arg_pattern, shape_pattern);
}

DFPattern IsCallTIR(const String& name, TuplePattern var_args, Array<Array<PrimExpr>> oshapes) {
  Array<DFPattern> shape_patterns;
  shape_patterns.reserve(oshapes.size());
  for (auto shape : oshapes) shape_patterns.push_back(PrimArrPattern(std::move(shape)));

  return IsOp("relax.call_tir")(GlobalVarPattern(name), var_args,
                                IsTuple(std::move(shape_patterns)));
}

DFPattern IsTuple(const Array<DFPattern>& fields) { return TuplePattern(fields); }
DFPattern IsTupleGetItem(const DFPattern tuple, int index) {
  return TupleGetItemPattern(tuple, index);
}

DFPattern DFPattern::dup() const {
  auto pattern = DFPatternDuplicator().VisitDFPattern(*this);
  return pattern;
}

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.dup_pattern").set_body_typed([](DFPattern pattern) {
  return pattern.dup();
});

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.dup_ubseq").set_body_typed([](UsedBySeq ub_seq) {
  return ub_seq.dup();
});

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.dup_oubseq").set_body_typed([](OnlyUsedBySeq oub_seq) {
  return oub_seq.dup();
});

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.PatternContext").set_body_typed([] {
  return PatternContext(make_object<PatternContextNode>());
});

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.current_context").set_body_typed([] {
  return PatternContext::Current();
});

class PatternContext::Internal {
 public:
  static void EnterScope(PatternContext pass_ctx) { pass_ctx.EnterWithScope(); }
  static void ExitScope(PatternContext pass_ctx) { pass_ctx.ExitWithScope(); }
};

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.enter_context")
    .set_body_typed(PatternContext::Internal::EnterScope);

TVM_REGISTER_GLOBAL("relax.dataflow_pattern.exit_context")
    .set_body_typed(PatternContext::Internal::ExitScope);

}  // namespace relax
}  // namespace tvm
