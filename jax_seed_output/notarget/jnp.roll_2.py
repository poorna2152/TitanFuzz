"""
1

"""
import jax
import jax.numpy as jnp
(TMAX, N) = (10, 256)
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (N,), dtype=jnp.float32)
b = jnp.zeros((N,), dtype=jnp.float32)

@jax.jit
def jacobi_1d(a):
    '1-D Jacobi stencil computation.'