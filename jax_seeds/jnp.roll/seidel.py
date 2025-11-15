import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Grid size
TMAX, N = 10, 128

# Initialize grid
key = jax.random.PRNGKey(0)
grid = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def seidel(grid):
    """2-D Seidel stencil computation."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        grid_new = (jnp.roll(grid, 1, axis=0) + jnp.roll(grid, -1, axis=0) +
                    jnp.roll(grid, 1, axis=1) + jnp.roll(grid, -1, axis=1)) / 4
        grid = grid.at[1:-1, 1:-1].set(grid_new[1:-1, 1:-1])
    return grid

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

stablehlo_seidel = export.export(seidel)(*input_shapes).mlir_module()
print(stablehlo_seidel)
