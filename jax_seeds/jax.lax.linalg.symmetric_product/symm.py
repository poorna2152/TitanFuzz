import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N = 512

# Initialize matrices and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
y = jnp.zeros((N,), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def symm(A, B, x, alpha, beta):
    y = jax.lax.linalg.symmetric_product(A, B, x, alpha, beta)
    return y

generate_stablehlo_and_export_metadata(symm, A, B, y, alpha, beta)