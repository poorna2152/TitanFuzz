import jax
import jax.numpy as jnp
import json
N = 512
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
def gemver(A, u1, v1):
    'Gemver kernel with 3 arguments, y calculated inside.'
    A = (A + jnp.outer(u1, v1))
    y = jnp.ones(A.shape[0], dtype=A.dtype)
    x = jnp.matmul(A, y)
    return x
input_shapes = [jax.ShapeDtypeStruct((N, N), jnp.float32), jax.ShapeDtypeStruct((N,), jnp.float32), jax.ShapeDtypeStruct((N,), jnp.float32)]
stablehlo_gemver = export.export(gemver)(*input_shapes).mlir_module()
print(stablehlo_gemver)

def generate_metadata(*args, func=None):
    args_meta = []
    for x in args:
        shape = list(x.shape)
        dtype = ('matrix' if (len(shape) > 1) else 'vector')
        args_meta.append({'type': dtype, 'shape': shape})
    metadata = {'args': args_meta}
    if (func is not None):
        output_shape_dtype = jax.eval_shape(func, *args)
        metadata['output'] = {'type': ('matrix' if (len(output_shape_dtype.shape) > 1) else 'vector'), 'shape': list(output_shape_dtype.shape)}
    filename = Path(__file__).name.replace('.py', '')
    pathname = (((filename + '/') + filename) + '.json')
    with open(pathname, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata
generate_metadata(A, u1, v1, func=gemver)