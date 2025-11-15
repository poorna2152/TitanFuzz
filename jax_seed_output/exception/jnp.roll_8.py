"""
{"exception": "TypeError", "msg": "PRNGKey() got multiple values for argument 'seed'"}
"""

import jax
import jax.numpy as jnp
(TMAX, N, M) = (10, 128, 128)
key = jax.random.PRNGKey(0, seed=88)
a = jax.random.uniform(key, (N, M), dtype=jnp.float32)

def jacobi_2d(a):
    '2-D Jacobi stencil computation.'
    for t in range(TMAX):
        b_new = ((((jnp.roll(a, 1, axis=0) + jnp.roll(a, (- 1), axis=0)) + jnp.roll(a, 1, axis=1)) + jnp.roll(a, (- 1), axis=1)) / 4)
