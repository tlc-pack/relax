
# Type annotation tests
@relax
def my_test(x : Tensor[_, _]):
    return None

@relax
def my_test(x: Tensor[(1, 2, 3), "int32"]):
    return None

@relax
def my_test(x: Tensor[(1, 2, 3), _]):
    return None

@relax
def my_test(x: Tensor[_, "int32"]):
    return None

# Builtin functions
@relax
def my_test(x: Tensor[_, "float32"]):
    match_shape(x, (1, 2, 3))