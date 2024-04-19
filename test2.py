import jax
import numpy as np
import jax.numpy as jnp
from functools import partial


def foo(x):
    return x * x + 1

data = jnp.array([1,2,3,4,5])

v_foo = jax.vmap(foo)

result = v_foo(data)

# Define a function that takes three arguments
def multiply_three_numbers(x, y, z):
    return x * y * z

# Create a new function with the first argument pre-specified
multiply_by_two_numbers = partial(multiply_three_numbers, 2)

# Call the new function with the remaining two arguments
result = multiply_by_two_numbers(3, 2)
print(result) # Output: 24


