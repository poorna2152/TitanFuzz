import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

N, M = 512, 512

key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, M), dtype=jnp.float32)
C = jax.random.uniform(key, (N, M), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def symm(A, B, C):
    alpha = 1.5
    beta = 1.2
    C = alpha * jnp.matmul(A, B) + beta * C
    return C

generate_stablehlo_and_export_metadata(symm, A, B, C)