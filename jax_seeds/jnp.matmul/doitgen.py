import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Tensor sizes
NQ, NR, NP = 128, 128, 128

# Initialize tensors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (NQ, NR, NP), dtype=jnp.float32)
C4 = jax.random.uniform(key, (NP, NP), dtype=jnp.float32)

@jax.jit
def doitgen(A, C4):
    """Performs the doitgen kernel."""
    for r in range(NR):
        A = A.at[:, r, :].set(jnp.matmul(A[:, r, :], C4))
    return A

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((NQ, NR, NP), jnp.float32),
    jax.ShapeDtypeStruct((NP, NP), jnp.float32),
]

stablehlo_doitgen = export.export(doitgen)(*input_shapes).mlir_module()
print(stablehlo_doitgen)