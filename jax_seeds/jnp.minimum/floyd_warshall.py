import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

N = 128

key = jax.random.PRNGKey(0)
dist = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def floyd_warshall(dist):
    for k in range(N):
        dist = jnp.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    return dist

generate_stablehlo_and_export_metadata(floyd_warshall, dist)
