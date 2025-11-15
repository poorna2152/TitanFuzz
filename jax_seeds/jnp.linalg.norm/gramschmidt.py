import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N, M = 128, 128

# Initialize matrix
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def gramschmidt(A):
    """Gram-Schmidt decomposition."""
    Q = jnp.zeros_like(A)
    for k in range(M):
        Q = Q.at[:, k].set(A[:, k])
        for i in range(k):
            r = jnp.dot(Q[:, i], Q[:, k])
            Q = Q.at[:, k].set(Q[:, k] - r * Q[:, i])
        norm = jnp.linalg.norm(Q[:, k])
        Q = Q.at[:, k].set(Q[:, k] / norm)
    return Q

generate_stablehlo_and_export_metadata(gramschmidt, A)