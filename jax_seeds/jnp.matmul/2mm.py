import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

NI, NJ, NK, NL = 512, 512, 512, 512

A = jnp.arange(NI * NK, dtype=jnp.float32).reshape(NI, NK) / (NI * NK)
B = jnp.arange(NK * NJ, dtype=jnp.float32).reshape(NK, NJ) / (NK * NJ)
C = jnp.arange(NJ * NL, dtype=jnp.float32).reshape(NJ, NL) / (NJ * NL)

@jax.jit
def two_mm(A, B, C):
    """Performs the 2mm kernel."""
    D = jnp.zeros((NI, NL), dtype=jnp.float32)
    for _ in range(100):
        tmp = jnp.matmul(A, B)
        D = jnp.matmul(tmp, C)
    return D

generate_stablehlo_and_export_metadata(two_mm, A, B, C)
