
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
    x1 = (x1 + jnp.matmul(jnp.transpose(A), y1))
    x2 = (x2 + jnp.matmul(A, y2))
    x = (x1 + x2)
    y = (jnp.dot(x2, y1) + jnp.dot(x1, y2))
    return x.astype(jnp.float32)
mvt_ = jax.jit(mvt)
generate_stablehlo_and_export_metadata(mvt, A, y1, y2)
