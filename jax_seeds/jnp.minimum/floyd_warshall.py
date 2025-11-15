import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Graph size
N = 128

# Initialize distance matrix
key = jax.random.PRNGKey(0)
dist = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def floyd_warshall(dist):
    """Floyd-Warshall shortest path algorithm."""
    for k in range(N):
        dist = jnp.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
    return dist

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

stablehlo_floyd_warshall = export.export(floyd_warshall)(*input_shapes).mlir_module()
print(stablehlo_floyd_warshall)
