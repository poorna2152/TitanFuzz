import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N, M = 512, 512

# Initialize matrices
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, M), dtype=jnp.float32)
B = jax.random.uniform(key, (N, M), dtype=jnp.float32)
C = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def syr2k(A, B, C, alpha, beta):
    """Performs the syr2k kernel."""
    C = alpha * (jnp.matmul(A, B.T) + jnp.matmul(B, A.T)) + beta * C
    return C

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
    jax.ShapeDtypeStruct((N, M), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
]

stablehlo_syr2k = export.export(syr2k)(*input_shapes).mlir_module()
print(stablehlo_syr2k)