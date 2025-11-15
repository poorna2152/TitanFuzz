
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
        sum_ = jnp.dot(r[:k].at[:k].at[:k].at[:k], (beta * r[:k].at[:k].at[:k].at[:k]))
        sum_ = jnp.maximum(sum_, jnp.inf)
        alpha = ((- (r[k] + sum_)) / beta)
        y = y.at[:k].set((y[:k] + (alpha * y[:k][::(- 1)])))
        y = y.at[k].set(alpha)
    return y
