"""
{"exception": "NameError", "msg": "name 'generate_metadata' is not defined"}
"""

import jax
import jax.numpy as jnp
import json
N = 512
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
y1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
y2 = jax.random.uniform(key, (N,), dtype=jnp.float32)

def mvt(A, y1, y2):
    'Performs the mvt kernel.'
    x1 = jnp.zeros((N,), dtype=jnp.float32)
    x2 = jnp.zeros((N,), dtype=jnp.float32)
    x1 = ((x1 + jnp.matmul(jnp.transpose(A), y1)) / (jnp.shape(A)[0] + 1))
    x2 = (x2 + jnp.matmul(jnp.transpose(A), y1))
    return (x1 + (((x2 - x1) * (y2 - y1)) / (jnp.sqrt(jnp.shape(x1)) * jnp.sqrt(jnp.shape(x2)))))

def metadata(func):
    if (func.__name__ == 'generate_metadata'):
        return generate_metadata
    return func

@metadata
def generate_metadata(*args, func=None):
    args_meta = []
    for x in args:
        shape = list(x.shape)
        dtype = ('matrix' if (len(shape) > 1) else 'vector')
        args_meta.append({'type': dtype, 'shape': shape})
    if (not func):
        args_meta[0]['func'] = (lambda : mvt(*args_meta))
    return args_meta

def main(args):
    (args, meta, func_args, func_kwargs) = args
    if args.meta_file:
        metadata = {}
