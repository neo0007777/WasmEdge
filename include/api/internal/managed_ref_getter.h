// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The WasmEdge Authors

//===-- wasmedge/api/internal/managed_ref_getter.h - shared getter gate -===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file shares the managed-ref predicates of the standalone C APIs
/// between the WasmEdge C-API, the WASM C-API and the GC test suite, so the
/// test asserts against the functions the C-APIs actually call.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "common/errcode.h"
#include "common/spdlog.h"
#include "common/types.h"
#include "executor/executor.h"

#include <utility>

namespace WasmEdge {

// A standalone C-API table or global has no owning module and thus no type
// list, so AST::TypeMatcher::refTypeCanHoldGCObject's module-relative
// resolution of a concrete heap-type index is unavailable. Resolve what can be
// resolved from the ref type alone: an abstract managed ref type (any/eq/i31/
// struct/array/extern/exn) is true, abstract funcref/nullfuncref is (provably)
// false, and a concrete type index -- indistinguishable here from a
// function-type index -- conservatively resolves to true (the safe direction:
// it may in fact be a struct/array type).
inline bool resolveCanHoldManagedNoTypeList(const ValType &RT) noexcept {
  return RT.isRefType() && !(RT.isFuncRefType() && RT.isAbsHeapType());
}

// True iff \p RT is a slot type whose legacy borrowed getter
// (WasmEdge_GlobalInstanceGetValue / WasmEdge_TableInstanceGetData) must
// return a typed-null sentinel instead of the live value. \p CanHoldManaged is
// the slot's construction-time canHoldManaged() bit; these getters do not add
// the reference to HostRoots, so a concurrent collection could reclaim the
// object while the host still holds the returned WasmEdge_Value.
//
// externref is excluded even though canHoldManaged() is true for it: it is the
// host-facing reference primitive, and round-tripping a host pointer through
// it is a shipped C-API workflow. funcref never holds a managed object, so
// CanHoldManaged is already false for it. The restricted types are therefore
// any/eq/i31/struct/array/exn and concrete struct/array type indices.
inline bool isRestrictedManagedRefGetterType(bool CanHoldManaged,
                                             const ValType &RT) noexcept {
  return CanHoldManaged && !RT.isExternRefType();
}

// Shared body of the producer-bearing RETAINED getters
// (WasmEdge_GlobalInstanceGetValueRetained /
// WasmEdge_TableInstanceGetDataRetained). \p Read reads the slot (a
// bounds-checked table read may fail, hence Expect), \p DeclType is the slot's
// declared ref type, and \p Exec is the producer executor whose allocator
// roots the reference. Returns the reference and the type to hand back:
//
//   - A live GC-managed (struct/array) reference is pinned as a host boundary
//     root through Allocator::retainResult -- the same call Executor::invoke
//     makes for a returned GC ref -- so a concurrent collection cannot reclaim
//     it until WasmEdge_ExecutorReleaseRef. It is handed back typed with its
//     resolved ABSTRACT heap type so the by-value release path matches it just
//     as it matches an Invoke-returned ref.
//   - A null reference, funcref, and externref are returned as-is with no
//     retention. externref is excluded because its host-pointer round trip must
//     keep working, and an externalized GC object handed over as externref is
//     released only via releaseAllRefs -- retaining it here would leak a root
//     the host cannot name.
//
// The foreign/unattached controller check is done by the CALLER before calling
// this (it is instance-scoped: TableInstance/GlobalInstance::
// hasForeignAllocator), so this is only ever reached for an instance the
// producer owns or that is unattached.
//
// The calling host thread is not a registered mutator, so no handshake ever
// waits for it: between the slot read and retainResult a guest could clear
// the slot and run a full collection, freeing the object under the expand
// (a payload dereference) and leaving a dangling root. The whole read ->
// expand -> retain sequence therefore runs while holding the controller's
// exclusive-operation token: a cycle in flight has swept before we get it,
// and no cycle can start (collect() takes the token non-blocking and skips)
// until we release it. The token's OwnedGrowing label is the only one a
// queued waiter may hold (see Controller::beginExclusiveOp); a reader taking
// it stops the world for nobody -- it merely excludes collections and grows
// for the microseconds of the read.
template <typename ReadSlot>
inline Expect<std::pair<RefVariant, ValType>>
retainManagedRefFromSlot(Executor::Executor &Exec, const ValType &DeclType,
                         ReadSlot &&Read) noexcept {
  struct ExclusiveReadWindow {
    GC::Controller &Ctrl;
    uint64_t Gen = 0;
    bool Held = false;
    explicit ExclusiveReadWindow(GC::Controller &C) noexcept : Ctrl(C) {
      Held = Ctrl.beginExclusiveOp(
          GC::Controller::ExclusiveOwner::State::OwnedGrowing, Gen);
    }
    ~ExclusiveReadWindow() noexcept {
      if (Held) {
        Ctrl.endExclusiveOp(
            Gen, GC::Controller::ExclusiveOwner::State::OwnedGrowing);
      }
    }
    ExclusiveReadWindow(const ExclusiveReadWindow &) = delete;
    ExclusiveReadWindow &operator=(const ExclusiveReadWindow &) = delete;
  } Window(Exec.getController());
  if (!Window.Held) {
    // Refused only when the controller is closing: the producer is being torn
    // down, so there is nothing safe to hand back.
    spdlog::error(ErrCode::Value::Interrupted);
    return Unexpect(ErrCode::Value::Interrupted);
  }
  EXPECTED_TRY(RefVariant Ref, Read());
  if (!DeclType.isExternRefType() && !Ref.isNull()) {
    // Resolve the reference's runtime heap kind, mirroring Executor::invoke: a
    // concrete type-index reference (from struct.new / array.new) carries an
    // opaque, defining-module-relative index, so expand it to its abstract
    // heap type (structref/arrayref) before the retain decision. Executor is a
    // friend of the defining ModuleInstance; the C-API is not, so this goes
    // through Executor::expandGCRefType.
    const ValType RefType = Exec.expandGCRefType(Ref);
    if (RefType.isGCRefType()) {
      // Live managed struct/array ref: root it for the host through the SAME
      // call Executor::invoke uses (retainResult) and return it typed with its
      // resolved abstract heap type so WasmEdge_ExecutorReleaseRef can match
      // it.
      Exec.getAllocator().retainResult(Ref);
      return std::pair<RefVariant, ValType>(Ref, RefType);
    }
  }
  // Non-managed / externref / null: pass the value through with its declared
  // slot type, no retention (mirrors the legacy non-restricted getter path).
  return std::pair<RefVariant, ValType>(Ref, DeclType);
}

} // namespace WasmEdge
