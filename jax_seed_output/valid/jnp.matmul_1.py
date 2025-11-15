
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
N = 512
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
x = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jnp.zeros((N,), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def gesummv(A, B, x):
    'Performs the gesummv kernel.'
    alpha = 1.5
    beta = 1.2
    y = ((alpha * jnp.matmul(A, x)) + (beta * jnp.matmul(B, x)))
    return y
generate_stablehlo_and_export_metadata(gesummv, A, B, x)
