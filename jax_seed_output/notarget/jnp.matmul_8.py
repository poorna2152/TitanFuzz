"""
1

"""
import logging
import jax
import jax.numpy as jnp
import numpy as np
import json

def get_stablehlo_asm(module_str: str) -> str:
    'Returns a pretty-printed StableHLO module with truncated large constants.'
    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
key = jax.random.PRNGKey(0)
(M, K, N) = (512, 256, 128)
A = jax.random.uniform(key, (M, K), dtype=jnp.float32)
B = jax.random.uniform(key, (K, N), dtype=jnp.float32)