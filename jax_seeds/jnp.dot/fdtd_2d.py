import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Grid size
TMAX, NX, NY = 10, 256, 256

# Initialize fields
key = jax.random.PRNGKey(0)
ex = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)
ey = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)
hz = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)

@jax.jit
def fdtd_2d(ex, ey, hz):
    """2-D Finite Difference Time Domain Kernel."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        ex_update = ex - 0.5 * (hz - jnp.roll(hz, 1, axis=0))
        ey_update = ey - 0.5 * (hz - jnp.roll(hz, 1, axis=1))
        hz_update = hz - 0.7 * ((jnp.roll(ex_update, -1, axis=1) - ex_update) + (jnp.roll(ey_update, -1, axis=0) - ey_update))
        # Only update interior points as before
        ex = ex.at[1:, :].set(ex_update[1:, :])
        ey = ey.at[:, 1:].set(ey_update[:, 1:])
        hz = hz.at[:-1, :-1].set(hz_update[:-1, :-1])
    return ex, ey, hz

generate_stablehlo_and_export_metadata(fdtd_2d, ex, ey, hz)