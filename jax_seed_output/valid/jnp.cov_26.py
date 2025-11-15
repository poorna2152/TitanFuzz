
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(N, M) = (512, 512)
key = jax.random.PRNGKey(0)
data = jax.random.uniform(key, (N, M), dtype=jnp.float32)

@jax.jit
def covariance(data):
    cov = jnp.cov(data, rowvar=False)
    return cov
generate_stablehlo_and_export_metadata(covariance, data)
