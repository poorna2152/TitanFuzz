"""
1

"""
import jax
import jax.numpy as jnp
import json
N = 512
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
y1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y2 = jax.random.uniform(key, (N,), dtype=jnp.float32)

def mvt(A, y1, y2):
    'Performs the mvt kernel.'
    x1 = jnp.zeros((N,), dtype=jnp.float32)