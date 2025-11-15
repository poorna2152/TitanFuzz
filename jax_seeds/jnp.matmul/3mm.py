import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

# Matrix sizes
NI, NJ, NK, NL, NM = 512, 512, 512, 512, 512

key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (NI, NK), dtype=jnp.float32)
B = jax.random.uniform(key, (NK, NJ), dtype=jnp.float32)
C = jax.random.uniform(key, (NJ, NM), dtype=jnp.float32)
D = jax.random.uniform(key, (NM, NL), dtype=jnp.float32)
E = jnp.zeros((NI, NJ), dtype=jnp.float32)
F = jnp.zeros((NJ, NL), dtype=jnp.float32)

@jax.jit
def three_mm(A, B, C, D):
    """Performs the 3mm kernel."""
    E = jnp.matmul(A, B)
    F = jnp.matmul(C, D)
    G = jnp.matmul(E, F)
    return G

generate_stablehlo_and_export_metadata(three_mm, A, B, C, D)