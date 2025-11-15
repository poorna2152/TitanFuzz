import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Grid size
TMAX, N = 10, 256

# Initialize arrays
key = jax.random.PRNGKey(0)
a = jax.random.uniform(key, (N,), dtype=jnp.float32)
b = jnp.zeros((N,), dtype=jnp.float32)

@jax.jit
def jacobi_1d(a):
    """1-D Jacobi stencil computation."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        b_new = (jnp.roll(a, 1) + a + jnp.roll(a, -1)) / 3
        a = a.at[1:-1].set(b_new[1:-1])
    return a

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((N,), jnp.float32),
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

generate_metadata(a, func=jacobi_1d)
stablehlo_jacobi_1d = export.export(jacobi_1d)(*input_shapes).mlir_module()
print(stablehlo_jacobi_1d)
