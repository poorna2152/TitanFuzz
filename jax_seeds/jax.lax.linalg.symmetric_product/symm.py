import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 512

# Initialize matrices and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
y = jnp.zeros((N,), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def symm(A, B, x, alpha, beta):
    y = jax.lax.linalg.symmetric_product(A, B, x, alpha, beta)
    return y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
]

stablehlo_symm = export.export(symm)(*input_shapes).mlir_module()
print(stablehlo_symm)