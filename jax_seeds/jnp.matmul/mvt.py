import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

@jax.jit
def mvt(A, y1, y2):
    """Performs the mvt kernel."""
    x1 = jnp.zeros((A.shape[0],), dtype=jnp.float32)
    x2 = jnp.zeros((A.shape[0],), dtype=jnp.float32)
    x1 = x1 + jnp.matmul(jnp.transpose(A), y1)
    x2 = x2 + jnp.matmul(A, y2)
    x = x1 + x2
    return x

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

if __name__ == "__main__":
    # Matrix size
    N = 512

    # Initialize matrix and vectors
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
    y1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
    y2 = jax.random.uniform(key, (N,), dtype=jnp.float32)

    # Export to StableHLO
    input_shapes = [
        jax.ShapeDtypeStruct((N, N), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
    ]

    stablehlo_mvt = export.export(mvt)(*input_shapes).mlir_module()
    print(stablehlo_mvt)
    generate_metadata(A, y1, y2, func=mvt)
# x = mvt(A, x1, x2, y1, y2)
# print(x)
# print("Output shape:", x.shape)