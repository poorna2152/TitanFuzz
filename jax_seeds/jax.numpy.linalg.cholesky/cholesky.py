import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 512

# Initialize matrix
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
A = A @ A.T + N * jnp.eye(N)  # Make A symmetric positive definite

@jax.jit
def cholesky(A):
    C = jax.numpy.linalg.cholesky(A)
    return C

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

stablehlo_cholesky = export.export(cholesky)(*input_shapes).mlir_module()
print(stablehlo_cholesky)
