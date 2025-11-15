import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Matrix size
N = 512

# Initialize matrices and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
x = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jnp.zeros((N,), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def gesummv(A, B, x, alpha, beta):
    """Performs the gesummv kernel."""
    y = alpha * jnp.matmul(A, x) + beta * jnp.matmul(B, x)
    return y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
    jax.ShapeDtypeStruct((), jnp.float32),
]

stablehlo_gesummv = export.export(gesummv)(*input_shapes).mlir_module()
print(stablehlo_gesummv)