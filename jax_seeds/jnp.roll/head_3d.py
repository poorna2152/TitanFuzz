import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# 3D grid size
TMAX, NX, NY, NZ = 10, 64, 64, 64

# Initialize grid
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (NX, NY, NZ), dtype=jnp.float32)

@jax.jit
def head_3d(u):
    """Heat equation over 3D data domain."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        u_new = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                 jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) +
                 jnp.roll(u, 1, axis=2) + jnp.roll(u, -1, axis=2)) / 6
        # Only update interior points
        u = u.at[1:-1, 1:-1, 1:-1].set(u_new[1:-1, 1:-1, 1:-1])
    return u

generate_stablehlo_and_export_metadata(head_3d, u)