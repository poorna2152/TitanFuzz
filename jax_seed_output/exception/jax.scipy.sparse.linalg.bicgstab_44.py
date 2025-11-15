"""
{"exception": "TypeError", "msg": "dot requires ndarray or scalar arguments, got <class 'tuple'> at position 0."}
"""

import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, x)
    b = (b - jnp.dot(s, x.dot(A)))
    s = jnp.linalg.solve(A, b)
    return s
sbjob = jax.jit(bicg(A, b, x))
print(sbjob)
