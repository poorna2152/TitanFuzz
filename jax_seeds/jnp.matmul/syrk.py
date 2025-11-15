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
C = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def syrk(A, C, alpha, beta):
    """Performs the syrk kernel."""
    C = alpha * jnp.matmul(A, A.T) + beta * C
    return C

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
]

stablehlo_syrk = export.export(syrk)(*input_shapes).mlir_module()
print(stablehlo_syrk)