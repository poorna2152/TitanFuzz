
import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N = 32

@jax.jit
def trisolv(L, b):
    """Performs the trisolv kernel using JAX's triangular_solve."""
    b_reshaped = b.reshape(-1, 1)
    x = jax.lax.linalg.triangular_solve(L, b_reshaped, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False)
    return x.flatten()

key = jax.random.PRNGKey(0)
L_full = jax.random.uniform(key, (N, N), dtype=jnp.float32)
L = jnp.tril(L_full)
L = L.at[jnp.diag_indices(N)].set(1.0)  # Ensure diagonal is 1
b = jax.random.uniform(key, (N,), dtype=jnp.float32)

generate_stablehlo_and_export_metadata(trisolv, L, b)
