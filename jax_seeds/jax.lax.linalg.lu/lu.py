import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix size
N = 512

# Initialize matrix
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def lu(A):
    return jax.lax.linalg.lu(A)

generate_stablehlo_and_export_metadata(lu, A)