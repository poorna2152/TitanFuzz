
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def correlation(data):
    mean = jnp.mean(data, axis=0)
    variance = jnp.var(data, axis=0)
    stddev = jnp.std(data, axis=0)
    normalized_data = ((data - mean) / stddev)
    corr = (jnp.matmul(normalized_data.T, normalized_data) / N)
    return corr
generate_stablehlo_and_export_metadata(correlation, data)
