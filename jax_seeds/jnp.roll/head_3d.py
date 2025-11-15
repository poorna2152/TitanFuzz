import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# 3D grid size
TMAX, NX, NY, NZ = 10, 64, 64, 64

# Initialize grid
key = jax.random.PRNGKey(0)
u = jax.random.uniform(key, (NX, NY, NZ), dtype=jnp.float32)

@jax.jit
def head_3d(u):
    """Heat equation over 3D data domain."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        u_new = (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
                 jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) +
                 jnp.roll(u, 1, axis=2) + jnp.roll(u, -1, axis=2)) / 6
        # Only update interior points
        u = u.at[1:-1, 1:-1, 1:-1].set(u_new[1:-1, 1:-1, 1:-1])
    return u

# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((NX, NY, NZ), jnp.float32),
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

generate_metadata(u, func=head_3d)
stablehlo_head_3d = export.export(head_3d)(*input_shapes).mlir_module()
print(stablehlo_head_3d)
