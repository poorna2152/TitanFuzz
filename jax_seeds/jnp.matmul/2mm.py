import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix sizes
NI, NJ, NK, NL = 512, 512, 512, 512

# Initialize matrices
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (NI, NK), dtype=jnp.float32)
B = jax.random.uniform(key, (NK, NJ), dtype=jnp.float32)
C = jax.random.uniform(key, (NJ, NL), dtype=jnp.float32)
D = jnp.zeros((NI, NL), dtype=jnp.float32)

@jax.jit
def two_mm(A, B, C):
    """Performs the 2mm kernel."""
    tmp = jnp.matmul(A, B)
    D = jnp.matmul(tmp, C)
    return D

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((NI, NK), jnp.float32),
    jax.ShapeDtypeStruct((NK, NJ), jnp.float32),
    jax.ShapeDtypeStruct((NJ, NL), jnp.float32),
]

stablehlo_2mm = export.export(two_mm)(*input_shapes).mlir_module()
print(stablehlo_2mm)