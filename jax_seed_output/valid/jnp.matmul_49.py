
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
N = 128
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5

@jax.jit
def trmm(A, B):
    'Triangular matrix-multiply.'
    alpha = 1.5
    A_tri = jnp.tril(A)
    B = ((alpha * jnp.matmul(B, A_tri)) + jnp.matmul(A_tri, A_tri.T))
    beta = A_tri.trace()
    B_tri = ((beta * jnp.matmul(B, A.T)) + jnp.matmul(beta, A_tri.T))
    return (A_tri + B_tri)
