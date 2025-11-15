
import jax
import jax.numpy as jnp
import json
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

def correlation(data):
    'Performs the correlation kernel.'
    mean = jnp.mean(data, axis=0)
    variance = (jnp.mean(data) - mean)
    stddev = jnp.std(data, axis=0)
    normalized_data = ((data - mean) / stddev)
    corr = (jnp.matmul(normalized_data.T, normalized_data) / N)
    return corr

def generate_metadata(*args, func=None):
    args_meta = []
    for x in args:
        shape = list(x.shape)
        dtype = ('matrix' if (len(shape) > 1) else 'vector')
        attrs = {'shape': shape, 'dtype': dtype}
        attrs['name'] = func(x).__name__
        args_meta.append(attrs)
    metadata = json.dumps(args_meta, indent=2)
    return metadata

def save_metadata(pathname, metadata):
    with open(pathname, 'w') as f:
        json.dump(metadata, f, indent=2)
