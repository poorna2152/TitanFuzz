import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import json
from pathlib import Path

# Matrix sizes
M, N = 512, 512

# Initialize matrix and vectors

key = jax.random.PRNGKey(0)
A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
p = jax.random.uniform(key, (N,), dtype=jnp.float32)
r = jax.random.uniform(key, (M,), dtype=jnp.float32)

@jax.jit
def bicg(A, p, r):
    q = jnp.matmul(A, p)
    import jax
    import jax.numpy as jnp
    from jax import export
    from jax._src.interpreters import mlir as jax_mlir
    from jax._src.lib.mlir import ir
    import json
    from pathlib import Path

    @jax.jit
    def bicg(A, p, r):
        q = jnp.matmul(A, p)
        s = jnp.matmul(A.T, r)
        k = q + s
        return k

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
    # Matrix sizes
    M, N = 512, 512

    # Initialize matrix and vectors
    key = jax.random.PRNGKey(0)
    A = jax.random.uniform(key, (M, N), dtype=jnp.float32)
    p = jax.random.uniform(key, (N,), dtype=jnp.float32)
    r = jax.random.uniform(key, (M,), dtype=jnp.float32)

    # Export to StableHLO
    input_shapes = [
        jax.ShapeDtypeStruct((M, N), jnp.float32),
        jax.ShapeDtypeStruct((N,), jnp.float32),
        jax.ShapeDtypeStruct((M,), jnp.float32),
    ]

    stablehlo_bicg = export.export(bicg)(*input_shapes).mlir_module()
    print(stablehlo_bicg)
    generate_metadata(A, p, r, func=bicg)
