
import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, x, b)
    a = jnp.dot(s, np.conj(a))
    s = jnp.linalg.solve(a, b)
    b = jnp.dot(s, jnp.conj(b))
    s = jnp.linalg.solve(A, b)
    return jnp.dot(s, jnp.linalg.solve(A.T, x))

def kullback_leibler(A, B):
    (sA, sB) = jnp.linalg.svd(A)
