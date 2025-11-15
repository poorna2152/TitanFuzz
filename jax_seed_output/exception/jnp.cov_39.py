"""
{"exception": "TypeError", "msg": "'NoneType' object is not callable"}
"""

import jax
import jax.numpy as jnp
import json
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

def covariance(data):
    cov = jnp.cov(data, rowvar=False)
    return cov

def generate_metadata(*args, func=None):
    args_meta = []
    for (key, value) in dict(zip(args, func()).items()):
        args_meta.append(f'"{key}": {value}')
    return '; '.join(args_meta)
if True:
    metadata = generate_metadata()
    filename = Path(__file__).name.replace('.py', '')
    with open((filename + '_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
