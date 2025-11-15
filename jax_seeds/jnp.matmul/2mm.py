import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

NI, NJ, NK, NL = 512, 512, 512, 512

A = jnp.arange(NI * NK, dtype=jnp.float32).reshape(NI, NK) / (NI * NK)
B = jnp.arange(NK * NJ, dtype=jnp.float32).reshape(NK, NJ) / (NK * NJ)
C = jnp.arange(NJ * NL, dtype=jnp.float32).reshape(NJ, NL) / (NJ * NL)

@jax.jit
def two_mm(A, B, C):
    """Performs the 2mm kernel."""
    D = jnp.zeros((NI, NL), dtype=jnp.float32)
    for _ in range(100):
        tmp = jnp.matmul(A, B)
        D = jnp.matmul(tmp, C)
    return D

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
    input_shapes = [
        jax.ShapeDtypeStruct((NI, NK), jnp.float32),
        jax.ShapeDtypeStruct((NK, NJ), jnp.float32),
        jax.ShapeDtypeStruct((NJ, NL), jnp.float32),
    ]
    stablehlo_2mm = export.export(two_mm)(*input_shapes).mlir_module()
    print(stablehlo_2mm)
    generate_metadata(A, B, C, func=two_mm)
