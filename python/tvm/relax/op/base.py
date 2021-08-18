from ...ir import BaseFunc
from ..expr import ShapeExpr, Tuple
from . import _make

def call_dps(shape, func, args):
    assert isinstance(func, BaseFunc)
    if isinstance(shape, (list, tuple)):
        shape = ShapeExpr(shape)
    if isinstance(args, (list, tuple)):
        args = Tuple(args)
    return _make.call_dps(shape, func, args)
