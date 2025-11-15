import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Grid size
N = 512
T = 10

# Initialize grid
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def adi(u):
    """Performs the adi kernel."""
    for t in range(T):
        # Use jnp.roll for stencil computation
        u_new = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                 jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1)) / 4
        # Only update interior points
        u = u.at[1:-1, 1:-1].set(u_new[1:-1, 1:-1])
    return u

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

stablehlo_adi = export.export(adi)(*input_shapes).mlir_module()
print(stablehlo_adi)