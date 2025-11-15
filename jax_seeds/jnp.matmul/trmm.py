import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N = 128

# Initialize matrices
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5

@jax.jit
def trmm(A, B):
    """Triangular matrix-multiply."""
    alpha = 1.5
    # Use lower triangular part of A
    A_tri = jnp.tril(A)
    B = alpha * jnp.matmul(A_tri, B)
    return B

generate_stablehlo_and_export_metadata(trmm, A, B)
