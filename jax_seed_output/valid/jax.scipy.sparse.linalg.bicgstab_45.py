
import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A.dot(x), b, check_finite=False)
    return jnp.real(s)

def lin_comb_lsq(A, b, x):
    (m, n) = A.shape
    A = jax.scipy.sparse.spdiags(A, 0, m, n, format=jnp.sp)
    s = jnp.dot(s, j)
    s = jnp.linalg.solve(A, s)
    s = jax.scipy.sparse.linalg.spdiags(s, 0, m, n, format=jnp.sp)
    b = jnp.dot(s, jnp.conj(b))
    s = jnp.linalg.solve(A, b)
