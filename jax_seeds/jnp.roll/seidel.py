import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata

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

generate_stablehlo_and_export_metadata(seidel, grid)