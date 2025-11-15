import jax
import jax.numpy as jnp
import json
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, M), dtype=jnp.float32)
B = jax.random.uniform(key, (N, M), dtype=jnp.float32)
C = jax.random.uniform(key, (N, N), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def syr2k(A, B, C):
    'Performs the syr2k kernel.'
    alpha = 1.5
    beta = 1.2
    C = ((alpha * (jnp.matmul(A, B.T) + jnp.matmul(B, A.T))) + (beta * C))
    return C
input_shapes = [jax.ShapeDtypeStruct((N, M), jnp.float32), jax.ShapeDtypeStruct((N, M), jnp.float32), jax.ShapeDtypeStruct((N, N), jnp.float32)]
stablehlo_syr2k = export.export(syr2k)(*input_shapes).mlir_module()
print(stablehlo_syr2k)

def generate_metadata(*args, func=None):
    args_meta = []
    for x in args:
        shape = list(x.shape)
        dtype = ('matrix' if (len(shape) > 1) else 'vector')
        args_meta.append({'type': dtype, 'shape': shape})
    metadata = {'args': args_meta}
    if (func is not None):
        output_shape_dtype = jax.eval_shape(func, *args)
        metadata['output'] = {'type': ('matrix' if (len(output_shape_dtype.shape) > 1) else 'vector'), 'shape': list(output_shape_dtype.shape), 'type': ('matrix' if (len(output_shape_dtype.shape) > 1) else 'vector')}
    filename = Path(__file__).name.replace('.py', '')
    pathname = (((filename + '/') + filename) + '.json')
    with open(pathname, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata
generate_metadata(A, B, C, func=syr2k)