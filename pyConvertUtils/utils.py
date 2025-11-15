import jax
import json
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
import logging
logging.disable(logging.WARNING)

def generate_metadata(func, *args):
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
    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(large_elements_limit=20)

def generate_stablehlo_and_export_metadata(func, *args):
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args]
    stablehlo_module = export.export(func)(*input_shapes).mlir_module()
    stablehlo_asm = get_stablehlo_asm(stablehlo_module)
    print(stablehlo_asm)
    generate_metadata(func, *args)
