import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Grid size
TMAX, N, M = 10, 128, 128

# Initialize arrays
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (N, M), dtype=jnp.float32)
b = jnp.zeros((N, M), dtype=jnp.float32)

@jax.jit
def jacobi_2d(a):
    """2-D Jacobi stencil computation."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        b_new = (jnp.roll(a, 1, axis=0) + jnp.roll(a, -1, axis=0) +
                 jnp.roll(a, 1, axis=1) + jnp.roll(a, -1, axis=1)) / 4
        a = a.at[1:-1, 1:-1].set(b_new[1:-1, 1:-1])
    return a

generate_stablehlo_and_export_metadata(jacobi_2d, a)