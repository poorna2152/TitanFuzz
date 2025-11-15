"""
1

"""
import jax
import jax.numpy as jnp
(TMAX, NX, NY, NZ) = (10, 64, 64, 64)
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (NX, NY, NZ), dtype=jnp.float32)

@jax.jit
def head_3d(u):
    'Heat equation over 3D data domain.'