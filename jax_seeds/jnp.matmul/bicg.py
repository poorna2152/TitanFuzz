import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix sizes
M, N = 512, 512

# Initialize matrix and vectors

key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
p = jax.random.uniform(key, (N,), dtype=jnp.float32)
r = jax.random.uniform(key, (M,), dtype=jnp.float32)

@jax.jit
def bicg(A, p, r):
    q = jnp.matmul(A, p)
    s = jnp.matmul(A.T, r)
    k = q + s
    return k

generate_stablehlo_and_export_metadata(bicg, A, p, r)