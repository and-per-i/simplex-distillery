import os
os.environ["JAX_PLATFORMS"] = "METAL"
import jax
import jax.numpy as jnp
print(f"Dispositivi JAX: {jax.devices()}")
try:
    x = jnp.ones((3, 3))
    print("Operazione semplice riuscita:")
    print(x + 1)
    
    key = jax.random.PRNGKey(0)
    print("PRNGKey riuscita.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
