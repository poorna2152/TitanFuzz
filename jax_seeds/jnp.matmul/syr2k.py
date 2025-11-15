import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

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
def syr2k(A, B, C):
    """Performs the syr2k kernel."""
    alpha = 1.5
    beta = 1.2
    C = alpha * (jnp.matmul(A, B.T) + jnp.matmul(B, A.T)) + beta * C
    return C

generate_stablehlo_and_export_metadata(syr2k, A, B, C)