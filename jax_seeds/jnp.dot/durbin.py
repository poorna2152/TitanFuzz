import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Vector size
N = 256

# Initialize vector
key = jax.random.PRNGKey(0)
r = jax.random.uniform(key, (N,), dtype=jnp.float32)

@jax.jit
def durbin(r):
    """Toeplitz system solver (Durbin's algorithm)."""
    y = jnp.zeros_like(r)
    beta = 1.0
    alpha = -r[0]
    y = y.at[0].set(alpha)
    for k in range(1, N):
        beta = beta * (1.0 - alpha ** 2)
        sum_ = jnp.dot(r[1:k+1][::-1], y[:k])
        alpha = -(r[k] + sum_) / beta
        y = y.at[:k].set(y[:k] + alpha * y[:k][::-1])
        y = y.at[k].set(alpha)
    return y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_durbin = export.export(durbin)(*input_shapes).mlir_module()
print(stablehlo_durbin)
