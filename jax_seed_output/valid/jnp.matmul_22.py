
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
N = 512
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
u1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
u2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
v1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
v2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jax.random.uniform(key, (N,), dtype=jnp.float32)
z = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((N,), dtype=jnp.float32)
w = jnp.zeros((N,), dtype=jnp.float32)

@jax.jit
def gemver(A, u1, v1):
    'Gemver kernel with 3 arguments, y calculated inside.'
    A = (A + jnp.outer(u1, v1))
    y = jnp.ones(A.shape[0], dtype=A.dtype)
    x = jnp.matmul(A, y)
