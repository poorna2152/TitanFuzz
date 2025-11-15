
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    mean = jnp.mean(data, axis=0)
    var = jnp.var(data, axis=0)
    data_norm = ((data - mean) / jnp.sqrt(var))
    corr_norm = ((data_norm * data_norm).sum(axis=0) / N)
    corr = (jnp.dot(corr_norm, data_norm.T) / N)
    stddev = jnp.std(data, axis=0)
    normalized_data = ((data - mean) / stddev)
    corr = (jnp.matmul(normalized_data.T, normalized_data) / N)
    return corr
generate_stablehlo_and_export_metadata(correlation, data)
