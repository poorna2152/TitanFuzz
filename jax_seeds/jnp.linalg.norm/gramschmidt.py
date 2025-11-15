import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

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

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
]

stablehlo_gramschmidt = export.export(gramschmidt)(*input_shapes).mlir_module()
print(stablehlo_gramschmidt)
