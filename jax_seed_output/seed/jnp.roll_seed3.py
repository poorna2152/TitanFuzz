import jax
import jax.numpy as jnp
(TMAX, N) = (10, 128)
key = jax.random.PRNGKey(0)
grid = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def seidel(grid):
    '2-D Seidel stencil computation.'
    for t in range(TMAX):
        grid_new = ((((jnp.roll(grid, 1, axis=0) + jnp.roll(grid, (- 1), axis=0)) + jnp.roll(grid, 1, axis=1)) + jnp.roll(grid, (- 1), axis=1)) / 4)