import jax
import jax.numpy as jnp
import json
(NI, NJ, NK, NL, NM) = (512, 512, 512, 512, 512)
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (NI, NK), dtype=jnp.float32)
B = jax.random.uniform(key, (NK, NJ), dtype=jnp.float32)
C = jax.random.uniform(key, (NJ, NM), dtype=jnp.float32)
D = jax.random.uniform(key, (NM, NL), dtype=jnp.float32)
E = jnp.zeros((NI, NJ), dtype=jnp.float32)
F = jnp.zeros((NJ, NL), dtype=jnp.float32)

@jax.jit
def three_mm(A, B, C, D):
    'Performs the 3mm kernel.'
    E = jnp.matmul(A, B)
    F = jnp.matmul(C, D)
    G = jnp.matmul(E, F)
    return G
input_shapes = [jax.ShapeDtypeStruct((NI, NK), jnp.float32), jax.ShapeDtypeStruct((NK, NJ), jnp.float32), jax.ShapeDtypeStruct((NJ, NM), jnp.float32), jax.ShapeDtypeStruct((NM, NL), jnp.float32)]
stablehlo_3mm = export.export(three_mm)(*input_shapes).mlir_module()
print(stablehlo_3mm)

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
generate_metadata(A, B, C, D, func=three_mm)