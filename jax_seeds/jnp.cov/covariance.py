import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N, M = 512, 512

# Initialize matrix
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def covariance(data):
    cov = jnp.cov(data, rowvar=False)
    return cov

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
]

stablehlo_covariance = export.export(covariance)(*input_shapes).mlir_module()
print(stablehlo_covariance)