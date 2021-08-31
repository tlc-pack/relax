from . import _ffi_api
from ..expr import Expr


def add(lhs: Expr,
        rhs: Expr) -> Expr:
    return _ffi_api.add(lhs, rhs)


def multiply(lhs: Expr,
             rhs: Expr) -> Expr:
    return _ffi_api.multiply(lhs, rhs)
