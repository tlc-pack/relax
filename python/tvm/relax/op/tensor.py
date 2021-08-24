from . import _ffi_api


def add(lhs, rhs):
    return _ffi_api.relax_add(lhs, rhs)


def multiply(lhs, rhs):
    return _ffi_api.relax_multiply(lhs, rhs)
