import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Grid size
TMAX, N, M = 10, 128, 128

# Initialize arrays
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (N, M), dtype=jnp.float32)
b = jnp.zeros((N, M), dtype=jnp.float32)

@jax.jit
def jacobi_2d(a):
    """2-D Jacobi stencil computation."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        b_new = (jnp.roll(a, 1, axis=0) + jnp.roll(a, -1, axis=0) +
                 jnp.roll(a, 1, axis=1) + jnp.roll(a, -1, axis=1)) / 4
        a = a.at[1:-1, 1:-1].set(b_new[1:-1, 1:-1])
    return a

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, M), jnp.float32),
]

stablehlo_jacobi_2d = export.export(jacobi_2d)(*input_shapes).mlir_module()
print(stablehlo_jacobi_2d)
