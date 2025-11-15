
import jax
import jax.numpy as jnp
from pyConvertUtils.utils import generate_stablehlo_and_export_metadata
(W, H) = (256, 256)
key = jax.random.PRNGKey(0)
img = jax.random.uniform(key, (H, W), dtype=jnp.float32)

@jax.jit
def deriche(img):
    Kx = jnp.array([[1, 0, (- 1)], [2, 0, (- 2)], [1, 0, (- 1)]], dtype=jnp.float32)
    Ky = jnp.array([[1, 2, 1], [0, 0, 0], [(- 1), (- 2), (- 1)]], dtype=jnp.float32)
    gx = jax.scipy.signal.convolve2d(img, Kx, mode='same')
    gy = jax.scipy.signal.convolve2d(img, Ky, mode='same')
    g = jnp.stack((gx, gy), axis=(- 1))
    edge = jnp.sqrt(((gx ** 2) + (gy ** 2)))
    return edge
generate_stablehlo_and_export_metadata(deriche, img)
