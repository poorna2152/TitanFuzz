import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix sizes
M, N = 512, 512

# Initialize matrix and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

@jax.jit
def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x)
    return s

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((M, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((M,), jnp.float32),
]

stablehlo_bicg = export.export(bicg)(*input_shapes).mlir_module()
print(stablehlo_bicg)
