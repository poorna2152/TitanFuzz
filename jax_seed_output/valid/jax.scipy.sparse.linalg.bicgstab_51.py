
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

@jax.jit
def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, A, b)
    s = (s - jnp.mean(s))
    s += jnp.mean(x, 1)
    s += jnp.mean((x * x), 1)
    s += jnp.mean(((x ** 2.0) * x), 1)
    return s

def compute_hlo_and_optimize(args):
    (M, N) = args
    A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
    b = jax.random.uniform
