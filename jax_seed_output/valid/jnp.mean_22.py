
import jax
import jax.numpy as jnp
import json
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

def correlation(data):
    'Performs the correlation kernel.'
    mean = jnp.mean(data, axis=0)
    variance = jnp.mean((data - mean), axis=0)
    stddev = jnp.sqrt(variance)
    return ((stddev ** 2) / N)

def correlation_kernel(data):
    'Performs the correlation kernel.'
