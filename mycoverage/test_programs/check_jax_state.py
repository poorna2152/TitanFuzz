import jax
import jax.numpy as jnp

from pyConvertUtils.utils import generate_stablehlo_and_export_metadata


def check_jax_state():
    """Simple function to check JAX state."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (3, 3))
    y = jnp.clip(x, a_min=0.0, a_max=1.0)

    print("Input Tensor:")
    print(x)
    print("Clipped Tensor:")
    print(y)

if __name__ == "__main__":
    check_jax_state()