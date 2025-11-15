import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

@jax.jit
def symm(A, B, C):
    alpha = 1.5
    beta = 1.2
    # Example symmetric matrix multiply: y = alpha * A @ B @ C + beta * C
    y = alpha * jnp.matmul(A, jnp.matmul(B, C)) + beta * C
    return y

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

    # Initialize matrices and vectors
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
    B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
    C = jax.random.uniform(key, (N,), dtype=jnp.float32)
    alpha = 1.5
    beta = 1.2

    # Export to StableHLO
    input_shapes = [
        jax.ShapeDtypeStruct((N, N), jnp.float32),
        jax.ShapeDtypeStruct((N, N), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32)
    ]

    stablehlo_symm = export.export(symm)(*input_shapes).mlir_module()
    print(stablehlo_symm)
    generate_metadata(A, B, C, func=symm)