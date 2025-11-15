import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

key = jax.random.PRNGKey(0)
M, K, N = 512, 256, 128

A = jax.random.uniform(key, (M, K), dtype=jnp.float32)
B = jax.random.uniform(key, (K, N), dtype=jnp.float32)

@jax.jit
def gemm(A, B):
    C = jnp.matmul(A, B)
    return C

generate_stablehlo_and_export_metadata(gemm, A, B)
