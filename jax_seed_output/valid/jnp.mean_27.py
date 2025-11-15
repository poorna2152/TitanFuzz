
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    mean = jnp.mean(data, axis=0)
    variance = jnp.sqrt(jnp.mean(((data - mean) ** 2), axis=0))
    stddev = jnp.clip(jnp.sqrt(variance), 1e-06, 1.0)
    return (- stddev)
