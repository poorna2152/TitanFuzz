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

@jax.jit
def lu(A):
    return jax.lax.linalg.lu(A)

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

stablehlo_lu = export.export(lu)(*input_shapes).mlir_module()
print(stablehlo_lu)