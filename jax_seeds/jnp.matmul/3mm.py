import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix sizes
NI, NJ, NK, NL, NM = 512, 512, 512, 512, 512

# Initialize matrices
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

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((NI, NK), jnp.float32),
    jax.ShapeDtypeStruct((NK, NJ), jnp.float32),
    jax.ShapeDtypeStruct((NJ, NM), jnp.float32),
    jax.ShapeDtypeStruct((NM, NL), jnp.float32),
]

stablehlo_3mm = export.export(three_mm)(*input_shapes).mlir_module()
print(stablehlo_3mm)