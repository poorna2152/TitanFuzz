
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
N = 256
key = jax.random.PRNGKey(0)
r = jax.random.uniform(key, (N,), dtype=jnp.float32)

@jax.jit
def durbin(r):
    "Toeplitz system solver (Durbin's algorithm)."
    y = jnp.zeros_like(r)
    beta = 1.0
    alpha = (- r[0])
    y = y.at[0].set(alpha)
    for k in range(1, N):
        beta = (beta * (1.0 - (alpha ** 2)))
        sum_ = jnp.dot(beta, y)
        sum_ = jnp.sum(sum_)
        alpha = ((- (r[k] + sum_)) / beta)
        y = y.at[:k].set((y[:k] + (alpha * y[:k][::(- 1)])))
        y = y.at[k].set(alpha)
    return y
generate_stablehlo_and_export_metadata(durbin, r)
