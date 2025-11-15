
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
N = 512
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
y1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y2 = jax.random.uniform(key, (N,), dtype=jnp.float32)

@jax.jit
def mvt(A, y1, y2):
    'Performs the mvt kernel.'
    x1 = jnp.zeros((N,), dtype=jnp.float32)
    x2 = jnp.zeros((N,), dtype=jnp.float32)
    x1 = (x1 + jnp.matmul(A.T, (y1 * y1).T))
    x2 = (x2 + jnp.matmul(A.T, (y2 * y2).T))
    x2 = (x2 / (jnp.sqrt(jnp.mean((x2 ** 2), axis=tuple(range(1, len(x1.shape))))) ** 0.5)).reshape((N, 1))
    return x1
