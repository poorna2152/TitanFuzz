import jax
import jax.numpy as jnp
import json
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def covariance(data):
    cov = jnp.cov(data, rowvar=False)
    return cov
input_shapes = [jax.ShapeDtypeStruct((N, M), jnp.float32)]
stablehlo_covariance = export.export(covariance)(*input_shapes).mlir_module()
print(stablehlo_covariance)

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
generate_metadata(data, func=covariance)