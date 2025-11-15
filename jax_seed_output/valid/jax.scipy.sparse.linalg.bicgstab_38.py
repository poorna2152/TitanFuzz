
import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(jnp.dot(s, jnp.conj(A)), x, b)
    s = jax.scipy.sparse.linalg.bicg(jnp.dot(s, jnp.conj(A)), s)
    s = jax.scipy.sparse.linalg.bicgstab(jnp.dot(s, jnp.conj(A)), s)
    b = jnp.dot(s, jnp.conj(b))
    s = jnp.linalg.solve(A, b)
