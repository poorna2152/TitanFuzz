import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Vector size
N = 256

# Initialize vector
key = jax.random.PRNGKey(0)
r = jax.random.uniform(key, (N,), dtype=jnp.float32)

@jax.jit
def durbin(r):
    """Toeplitz system solver (Durbin's algorithm)."""
    y = jnp.zeros_like(r)
    beta = 1.0
    alpha = -r[0]
    y = y.at[0].set(alpha)
    for k in range(1, N):
        beta = beta * (1.0 - alpha ** 2)
        sum_ = jnp.dot(r[1:k+1][::-1], y[:k])
        alpha = -(r[k] + sum_) / beta
        y = y.at[:k].set(y[:k] + alpha * y[:k][::-1])
        y = y.at[k].set(alpha)
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

generate_metadata(r, func=durbin)
# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N,), jnp.float32),
]

stablehlo_durbin = export.export(durbin)(*input_shapes).mlir_module()
print(stablehlo_durbin)
