import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Grid size
TMAX, NX, NY = 10, 256, 256

# Initialize fields
key = jax.random.PRNGKey(0)
ex = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)
ey = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)
hz = jax.random.uniform(key, (NX, NY), dtype=jnp.float32)

@jax.jit
def fdtd_2d(ex, ey, hz):
    """2-D Finite Difference Time Domain Kernel."""
    for t in range(TMAX):
        # Use jnp.roll for stencil computation
        ex_update = ex - 0.5 * (hz - jnp.roll(hz, 1, axis=0))
        ey_update = ey - 0.5 * (hz - jnp.roll(hz, 1, axis=1))
        hz_update = hz - 0.7 * ((jnp.roll(ex_update, -1, axis=1) - ex_update) + (jnp.roll(ey_update, -1, axis=0) - ey_update))
        # Only update interior points as before
        ex = ex.at[1:, :].set(ex_update[1:, :])
        ey = ey.at[:, 1:].set(ey_update[:, 1:])
        hz = hz.at[:-1, :-1].set(hz_update[:-1, :-1])
    return ex, ey, hz

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

generate_metadata(ex, ey, hz, func=fdtd_2d)
# Export to StableHLO
input_shapes = [
    jax.ShapeDtypeStruct((NX, NY), jnp.float32),
    jax.ShapeDtypeStruct((NX, NY), jnp.float32),
    jax.ShapeDtypeStruct((NX, NY), jnp.float32),
]

stablehlo_fdtd_2d = export.export(fdtd_2d)(*input_shapes).mlir_module()
print(stablehlo_fdtd_2d)
