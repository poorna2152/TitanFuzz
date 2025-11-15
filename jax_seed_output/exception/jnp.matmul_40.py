"""
{"exception": "NameError", "msg": "name 'generate_metadata' is not defined"}
"""

import jax
import jax.numpy as jnp
import json
N = 128
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)

def trmm(A, B):
    'Triangular matrix-multiply.'
    alpha = 1.5
    A_tri = jnp.tril(A)
    B = ((alpha * jnp.matmul(B, A)) - (alpha * jnp.matmul(A_tri, B)))

def calculate_metrics(key, A, B):
    metadata = {'metric_type': 'mse', 'key_string': str(key), 'A': calculate_metrics(key, A, A), 'B': calculate_metrics(key, B, B)}
    return metadata
generate_metadata(A, B, func=trmm)
