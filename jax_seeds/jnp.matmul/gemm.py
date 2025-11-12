import logging
import jax
import jax.numpy as jnp
import numpy as np
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

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
