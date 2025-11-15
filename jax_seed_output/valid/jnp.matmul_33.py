
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
x = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jnp.zeros((M,), dtype=jnp.float32)

@jax.jit
def atax(A, x):
    'Performs the atax kernel.'
    y = jnp.matmul(A, x)
    return y
generate_stablehlo_and_export_metadata(atax, A, x)
