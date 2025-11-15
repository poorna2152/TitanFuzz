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

def generate_metadata(*args, func=None):
    args_meta = []
    for x in args:
        shape = list(x.shape)
        dtype = "matrix" if len(shape) > 1 else "vector"
        args_meta.append({"type": dtype, "shape": shape})

    metadata = {"args": args_meta}

    # --- Calculate output shape using jax.eval_shape ---
    if func is not None:
        output_shape_dtype = jax.eval_shape(func, *args)
        metadata["output"] = {
            "type": "matrix" if len(output_shape_dtype.shape) > 1 else "vector",
            "shape": list(output_shape_dtype.shape),
            "type": "matrix" if len(output_shape_dtype.shape) > 1 else "vector"
        }
    
    filename = Path(__file__).name.replace(".py", "")
    pathname = filename + "/" + filename + ".json"
    with open(pathname, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

generate_metadata(img, func=deriche)