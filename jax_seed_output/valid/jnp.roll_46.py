
import jax
import jax.numpy as jnp
N = 512
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (N, N), dtype=jnp.float32)

def adi(u):
    'Performs the adi kernel.'
    for t in range(T):
        u_new = ((((jnp.roll(u, (- 3), axis=0) + jnp.roll(u, (- 1), axis=0)) + jnp.roll(u, 1, axis=1)) + jnp.roll(u, (- 1), axis=1)) / 4)
        u = u_new
    return u
