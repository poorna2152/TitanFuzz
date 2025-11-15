import jax
import jax.numpy as jnp
import json
(M, N) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
b = jax.random.uniform(key, (N,), dtype=jnp.float32)
x = jnp.zeros((M,), dtype=jnp.float32)

@jax.jit
def bicg(A, b, x):
    s = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x)
    return s
input_shapes = [jax.ShapeDtypeStruct((M, N), jnp.float32), jax.ShapeDtypeStruct((N,), jnp.float32), jax.ShapeDtypeStruct((M,), jnp.float32)]
stablehlo_bicg = export.export(bicg)(*input_shapes).mlir_module()
print(stablehlo_bicg)