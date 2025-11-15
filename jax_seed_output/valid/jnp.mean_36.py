
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    mean = jnp.mean(data, axis=0)
    std = jax.math.softplus(jax.math.sqrt(jnp.mean(data, axis=0)))
    return (std / (((jnp.sqrt(2.0) - 1.0) * jnp.square((std - mean))) + (jnp.sqrt(2.0) * std)))
    variance = ((jnp.var(data, axis=0) + jnp.mean(data, axis=0)) - jax.mean(data, axis=0))
