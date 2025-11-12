import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix sizes
M, N = 512, 512

# Initialize matrix and vector
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
x = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jnp.zeros((M,), dtype=jnp.float32)

@jax.jit
def atax(A, x):
    """Performs the atax kernel."""
    y = jnp.matmul(jnp.matrix_transpose(A), jnp.matmul(A, x))
    return y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((M, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_atax = export.export(atax)(*input_shapes).mlir_module()
print(stablehlo_atax)