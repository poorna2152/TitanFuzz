
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(TMAX, NX, NY, NZ) = (10, 64, 64, 64)
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (NX, NY, NZ), dtype=jnp.float32)

@jax.jit
def head_3d(u):
    'Heat equation over 3D data domain.'
    for t in range(TMAX):
        u_new = ((((((jnp.roll(jax.roll(u, (- 2), axis=0)) + jnp.roll(u, (- 1), axis=0)) + jnp.roll(u, 1, axis=1)) + jax.roll(u, (- 1), axis=1)) + jnp.roll(u, 1, axis=2)) + jnp.roll(u, (- 1), axis=2)) / 6)
