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
#ifndef TVM_SCRIPT_IR_BUILDER_BASE_H_
#define TVM_SCRIPT_IR_BUILDER_BASE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/node/node.h>

#include <vector>

namespace tvm {
namespace script {
namespace ir_builder {

////////////////////////////// Core Infra: Frame //////////////////////////////

class IRBuilderFrameNode : public runtime::Object {
 public:
  std::vector<runtime::TypedPackedFunc<void()>> callbacks;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `callbacks` is not visited.
  }

  static constexpr const char* _type_key = "script.ir_builder.IRBuilderFrame";
  TVM_DECLARE_BASE_OBJECT_INFO(IRBuilderFrameNode, runtime::Object);

 public:
  virtual ~IRBuilderFrameNode() = default;
  virtual void EnterWithScope();
  virtual void ExitWithScope();

  void AddCallback(runtime::TypedPackedFunc<void()> callback);
};

class IRBuilderFrame : public runtime::ObjectRef {
 public:
  virtual ~IRBuilderFrame() = default;
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRBuilderFrame, ObjectRef, IRBuilderFrameNode);

 protected:
  IRBuilderFrame() = default;

 public:
  inline void EnterWithScope();
  inline void ExitWithScope();
};

////////////////////////////// Core Infra: Builder //////////////////////////////
///
class IRBuilderNode : public runtime::Object {
 public:
  runtime::Array<IRBuilderFrame> frames;
  Optional<ObjectRef> result;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("frames", &frames);
    v->Visit("result", &result);
  }

  static constexpr const char* _type_key = "script.ir_builder.IRBuilder";
  TVM_DECLARE_FINAL_OBJECT_INFO(IRBuilderNode, runtime::Object);

 public:
  template <typename TFrame>
  inline Optional<TFrame> FindFrame() const;
  template <typename TFrame>
  inline Optional<TFrame> GetLastFrame() const;
  template <typename TObjectRef>
  inline TObjectRef Get() const;
};

class IRBuilder : public runtime::ObjectRef {
 public:
  IRBuilder();
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(IRBuilder, ObjectRef, IRBuilderNode);

 public:
  void EnterWithScope();
  void ExitWithScope();
  static IRBuilder Current();
  template <class TObjectRef>
  inline static TObjectRef Name(String name, TObjectRef obj);
};

////////////////////////////// Details //////////////////////////////

namespace details {

class Namer {
 public:
  using FType = NodeFunctor<void(const ObjectRef&, String)>;
  static FType& vtable();
  static void Name(ObjectRef node, String name);
};

}  // namespace details

template <class TObjectRef>
inline TObjectRef IRBuilder::Name(String name, TObjectRef obj) {
  details::Namer::Name(obj, name);
  return Downcast<TObjectRef>(obj);
}

inline void IRBuilderFrame::EnterWithScope() {
  ICHECK(data_ != nullptr);
  static_cast<IRBuilderFrameNode*>(data_.get())->EnterWithScope();
}

inline void IRBuilderFrame::ExitWithScope() {
  ICHECK(data_ != nullptr);
  static_cast<IRBuilderFrameNode*>(data_.get())->ExitWithScope();
  data_.reset();
}

template <typename TFrame>
inline Optional<TFrame> IRBuilderNode::FindFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
    if (const TFrameNode* p = (*it).template as<TFrameNode>()) {
      return GetRef<TFrame>(p);
    }
  }
  return NullOpt;
}

template <typename TFrame>
inline Optional<TFrame> IRBuilderNode::GetLastFrame() const {
  using TFrameNode = typename TFrame::ContainerType;
  if (!frames.empty() && frames.back()->IsInstance<TFrameNode>()) {
    return Downcast<TFrame>(frames.back());
  }
  return NullOpt;
}

template <typename TObjectRef>
inline TObjectRef IRBuilderNode::Get() const {
  using TObject = typename TObjectRef::ContainerType;
  CHECK(result.defined()) << "IndexError: No result exists in IRBuilder yet";
  const auto* n = result.as<TObject>();
  CHECK(n != nullptr) << "IndexError: IRBuilder result is not of type: " << TObject::_type_key;
  return GetRef<TObjectRef>(n);
}

}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_BASE_H_
