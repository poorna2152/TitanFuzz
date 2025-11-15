"""
1

"""
import jax
import jax.numpy as jnp
N = 512
T = 10
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def adi(u):
    'Performs the adi kernel.'