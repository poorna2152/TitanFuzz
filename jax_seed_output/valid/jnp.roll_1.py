
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(TMAX, N) = (10, 256)
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (N,), dtype=jnp.float32)
b = jnp.zeros((N,), dtype=jnp.float32)

@jax.jit
def jacobi_1d(a):
    '1-D Jacobi stencil computation.'
    for t in range(TMAX):
        b_new = (((jnp.roll(jnp.roll(a, (- 1)), (- 1)) + a) + jnp.roll(a, (- 1))) / 3)
        a = a.at[1:(- 1)].set(b_new[1:(- 1)])
    return a
generate_stablehlo_and_export_metadata(jacobi_1d, a)
