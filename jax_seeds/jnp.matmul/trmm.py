import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N = 128

# Initialize matrices
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5

@jax.jit
def trmm(A, B, alpha):
    """Triangular matrix-multiply."""
    # Use lower triangular part of A
    A_tri = jnp.tril(A)
    B = alpha * jnp.matmul(A_tri, B)
    return B

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
]

stablehlo_trmm = export.export(trmm)(*input_shapes).mlir_module()
print(stablehlo_trmm)
