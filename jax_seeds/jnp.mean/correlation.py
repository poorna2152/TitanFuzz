import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N, M = 512, 512

# Initialize matrix
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    """Performs the correlation kernel."""
    mean = jnp.mean(data, axis=0)
    stddev = jnp.std(data, axis=0)
    normalized_data = (data - mean) / stddev
    corr = jnp.matmul(normalized_data.T, normalized_data) / N
    return corr

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
]

stablehlo_correlation = export.export(correlation)(*input_shapes).mlir_module()
print(stablehlo_correlation)