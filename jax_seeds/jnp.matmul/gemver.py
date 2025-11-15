import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N = 512

# Initialize matrix and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
u1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
u2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
v1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
v2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jax.random.uniform(key, (N,), dtype=jnp.float32)
z = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((N,), dtype=jnp.float32)
w = jnp.zeros((N,), dtype=jnp.float32)

@jax.jit
def gemver(A, u1, u2, v1, v2, y, z):
    """Performs the gemver kernel."""
    A = A + jnp.outer(u1, v1) + jnp.outer(u2, v2)
    x = x + jnp.matmul(A, y)
    w = w + jnp.matmul(A.T, z)
    return x, w

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_gemver = export.export(gemver)(*input_shapes).mlir_module()
print(stablehlo_gemver)