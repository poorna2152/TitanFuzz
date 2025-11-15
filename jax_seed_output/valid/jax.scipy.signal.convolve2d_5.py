
import jax
import jax.numpy as jnp
import json
(W, H) = (256, 256)
key = jax.random.PRNGKey(0)
img = jax.random.uniform(key, (H, W), dtype=jnp.float32)

def deriche(img):
    'Simple edge detection filter (deriche).'
    Kx = jnp.array([[1, 0, (- 1)], [2, 0, (- 2)], [1, 0, (- 1)]], dtype=jnp.float32)
    Ky = jnp.array([[1, 2, 1], [0, 0, 0], [(- 1), (- 2), (- 1)]], dtype=jnp.float32)
    gx = jax.scipy.signal.convolve2d(img, Kx, mode='same')
    gy = jax.scipy.signal.convolve2d(img, Ky, mode='same')
