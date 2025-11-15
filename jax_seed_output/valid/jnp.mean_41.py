
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    mean = jnp.mean(data)
    sigma = jnp.std(data)
    correlation = jax.linalg.inv(sigma)
    correlation = jax.linalg.solve(correlation, jnp.transpose(data))
    correlation = jax.nn.relu(correlation)
    correlation = jax.exp(((- 1.0) * ((data / mean) ** 2)))
    return correlation
jitted_correlation = jax.jit(correlation)
