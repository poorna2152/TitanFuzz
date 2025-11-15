import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix size
N = 512

# Initialize matrices and vectors
key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (N, N), dtype=jnp.float32)
B = jax.random.uniform(key, (N, N), dtype=jnp.float32)
x = jax.random.uniform(key, (N,), dtype=jnp.float32)
y = jnp.zeros((N,), dtype=jnp.float32)
alpha = 1.5
beta = 1.2

@jax.jit
def gesummv(A, B, x):
    """Performs the gesummv kernel."""
    alpha = 1.5
    beta = 1.2
    y = alpha * jnp.matmul(A, x) + beta * jnp.matmul(B, x)
    return y

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N, N), jnp.float32),
    jax.ShapeDtypeStruct((N,), jnp.float32)
]

stablehlo_gesummv = export.export(gesummv)(*input_shapes).mlir_module()
print(stablehlo_gesummv)

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

generate_metadata(A, B, x, func=gesummv)