import logging
import jax
import jax.numpy as jnp
import numpy as np
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

logging.disable(logging.WARNING)

def get_stablehlo_asm(module_str: str) -> str:
    """Returns a pretty-printed StableHLO module with truncated large constants."""
    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
        return stablehlo_module.operation.get_asm(large_elements_limit=20)


# -----------------------------
# Matrix Initialization
# -----------------------------
key = jax.random.PRNGKey(0)
M, K, N = 512, 256, 128

A = jax.random.uniform(key, (M, K), dtype=jnp.float32)
B = jax.random.uniform(key, (K, N), dtype=jnp.float32)

@jax.jit
def gemm(A, B):
    C = jnp.matmul(A, B)
    return C

# -----------------------------
# Export to StableHLO
# -----------------------------
input_shapes = [
    jax.ShapeDtypeStruct((M, K), jnp.float32),
    jax.ShapeDtypeStruct((K, N), jnp.float32),
]

stablehlo_gemm = export.export(gemm)(*input_shapes).mlir_module()
print(get_stablehlo_asm(stablehlo_gemm))

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

generate_metadata(A, B, func=gemm)