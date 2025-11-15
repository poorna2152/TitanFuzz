import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

@jax.jit
def gemver(A, u1, v1):
    """Gemver kernel with 3 arguments, y calculated inside."""
    A = A + jnp.outer(u1, v1)
    # Calculate y as a vector of ones (can be changed as needed)
    y = jnp.ones(A.shape[0], dtype=A.dtype)
    x = jnp.matmul(A, y)
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
            "shape": list(output_shape_dtype.shape)
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
    u1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
    u2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
    v1 = jax.random.uniform(key, (N,), dtype=jnp.float32)
    v2 = jax.random.uniform(key, (N,), dtype=jnp.float32)
    y = jax.random.uniform(key, (N,), dtype=jnp.float32)
    z = jax.random.uniform(key, (N,), dtype=jnp.float32)
    x = jnp.zeros((N,), dtype=jnp.float32)
    w = jnp.zeros((N,), dtype=jnp.float32)

    input_shapes = [
        jax.ShapeDtypeStruct((N, N), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
    ]

    stablehlo_gemver = export.export(gemver)(*input_shapes).mlir_module()
    print(stablehlo_gemver)
    generate_metadata(A, u1, v1, func=gemver)