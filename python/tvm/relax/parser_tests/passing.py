from tvm.relax.parser import relax

# Type annotation tests
"""
@relax
def my_test(x : Tensor[_, _]):
    return None

@relax
def my_test(x: Tensor[(a, b, c), "int32"]):
    return None

@relax
def my_test(x: Tensor[(1, 2, 3), _]):
    return None

@relax
def my_test(x: Tensor[_, "int32"]):
    return None
"""
# Builtin functions

@relax
def my_test(x: Tensor[_, "float32"]):
    match_shape(x.shape, (1, 2, 3))


# These should pass in the future but don't right now
"""
@relax
def my_test(x: Tensor[_, "float32"]):
    match_shape(x.shape, (1, 2, 3))


@relax
def my_test(x : Tensor[(1, 2, 3), "int32"], y: Tensor[_, _]):
    return call_packed("my_func", x, y)
"""