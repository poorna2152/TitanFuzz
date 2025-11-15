"""
{"exception": "TypeError", "msg": "JAX encountered invalid PRNG key data: expected key_data to have ndim, shape, and dtype attributes. Got 0"}
"""

import jax
import jax.numpy as jnp
import json
N = 256
key = jax.random.bernoulli(0)
r = jax.random.uniform(key, (N,), dtype=jnp.float32)

def durbin(r):
    "Toeplitz system solver (Durbin's algorithm)."
    y = jnp.zeros_like(r)
    beta = 1.0
    alpha = (- r[0])
    y = y.at[0].set(alpha)
    for k in range(1, N):
        beta = (beta * (1.0 - (alpha ** 2)))
        sum_ = (jnp.dot(y, beta) / jnp.sum(y))
