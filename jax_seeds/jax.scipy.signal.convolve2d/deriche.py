import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Image size
W, H = 256, 256

# Initialize image
key = jax.random.PRNGKey(0)
img = jax.random.uniform(key, (H, W), dtype=jnp.float32)

@jax.jit
def deriche(img):
    """Simple edge detection filter (deriche)."""
    # Sobel filter as a placeholder for deriche
    Kx = jnp.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=jnp.float32)
    Ky = jnp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=jnp.float32)
    gx = jax.scipy.signal.convolve2d(img, Kx, mode='same')
    gy = jax.scipy.signal.convolve2d(img, Ky, mode='same')
    edge = jnp.sqrt(gx ** 2 + gy ** 2)
    return edge

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((H, W), jnp.float32),
]

stablehlo_deriche = export.export(deriche)(*input_shapes).mlir_module()
print(stablehlo_deriche)
