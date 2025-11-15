
import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)

def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, a, b)
    return s.toarray()
