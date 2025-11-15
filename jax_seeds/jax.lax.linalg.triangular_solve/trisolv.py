import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 32

# Initialize matrix and vector

key = jax.random.PRNGKey(0)
L_full = jax.random.uniform(key, (N, N), dtype=jnp.float32)
L = jnp.tril(L_full)
L = L.at[jnp.diag_indices(N)].set(1.0)  # Ensure diagonal is 1
b = jax.random.uniform(key, (N,), dtype=jnp.float32)


@jax.jit
def trisolv(L, b):
    """Performs the trisolv kernel using JAX's triangular_solve."""
    b_reshaped = b.reshape(-1, 1)
    x = jax.lax.linalg.triangular_solve(L, b_reshaped, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False)
    return x.flatten()

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_trisolv = export.export(trisolv)(*input_shapes).mlir_module()
print(stablehlo_trisolv)