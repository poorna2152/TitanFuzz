import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N = 512

# Initialize matrix and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
x1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
x2 = jnp.zeros((N,), dtype=jnp.float32)
y1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y2 = jax.random.uniform(key, (N,), dtype=jnp.float32)

@jax.jit
def mvt(A, x1, x2, y1, y2):
    """Performs the mvt kernel."""
    x1 = x1 + jnp.matmul(jnp.transpose(A), y1)
    x2 = x2 + jnp.matmul(A, y2)
    return x1, x2

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_mvt = export.export(mvt)(*input_shapes).mlir_module()
print(stablehlo_mvt)
