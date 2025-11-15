import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N = 512

# Initialize matrix
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
A = A @ A.T + N * jnp.eye(N)  # Make A symmetric positive definite

@jax.jit
def cholesky(A):
    C = jax.numpy.linalg.cholesky(A)
    return C

generate_stablehlo_and_export_metadata(cholesky, A)

