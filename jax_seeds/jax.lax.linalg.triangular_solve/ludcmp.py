import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 128

# Initialize matrix
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)

@jax.jit
def ludcmp(A):
    """LU decomposition followed by forward substitution."""
    L = jnp.eye(N, dtype=jnp.float32)
    U = A.copy()
    for i in range(N):
        for j in range(i+1, N):
            factor = U[j, i] / U[i, i]
            L = L.at[j, i].set(factor)
            U = U.at[j, :].set(U[j, :] - factor * U[i, :])
    # Forward substitution: solve L y = b (b = ones) using triangular_solve
    b = jnp.ones((N,), dtype=jnp.float32)
    y = jax.lax.linalg.triangular_solve(L, b.reshape(-1, 1), left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False)
    y = y.flatten()
    return L, U, y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
]

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


generate_metadata(A, func=ludcmp)
stablehlo_ludcmp = export.export(ludcmp)(*input_shapes).mlir_module()
print(stablehlo_ludcmp)
