
import jax
import jax.numpy as jnp
import json
N = 256
key = jax.random.PRNGKey(0)
r = jax.random.uniform(key, (N,), dtype=jnp.float32)

def durbin(r):
    "Toeplitz system solver (Durbin's algorithm)."
    y = jnp.zeros_like(r)
    beta = 1.0
    alpha = (- r[0])
    y = y.at[0].set(alpha)
    for k in range(1, N):
        beta = (beta * (1.0 - (alpha ** 2)))
        sum_ = (jnp.dot(y.at[:k], y[:k]) / beta)
        y = jnp.roll(y, ((- 1) * k), axis=0)
        y = jnp.dot(y, ((sum_ + jnp.dot(y, y)) / beta))
