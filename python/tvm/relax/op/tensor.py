from . import _ffi_api
from ..expr import Expr


def add(lhs: Expr,
        rhs: Expr) -> Expr:
    return _ffi_api.relax_add(lhs, rhs)


def multiply(lhs: Expr,
             rhs: Expr) -> Expr:
    return _ffi_api.relax_multiply(lhs, rhs)
