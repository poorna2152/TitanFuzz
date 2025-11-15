
import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 32

@jax.jit
def trisolv(L, b):
    """Performs the trisolv kernel using JAX's triangular_solve."""
    b_reshaped = b.reshape(-1, 1)
    x = jax.lax.linalg.triangular_solve(L, b_reshaped, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False)
    return x.flatten()

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
    # Initialize matrix and vector
    key = jax.random.PRNGKey(0)
    L_full = jax.random.uniform(key, (N, N), dtype=jnp.float32)
    L = jnp.tril(L_full)
    L = L.at[jnp.diag_indices(N)].set(1.0)  # Ensure diagonal is 1
    b = jax.random.uniform(key, (N,), dtype=jnp.float32)

    # Export to StableHLO
    input_shapes = [
        jax.ShapeDtypeStruct((N, N), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
    ]

    generate_metadata(L, b, func=trisolv)
    stablehlo_trisolv = export.export(trisolv)(*input_shapes).mlir_module()
    print(stablehlo_trisolv)