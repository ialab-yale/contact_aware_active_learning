import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad, lax

class SoftContactModel:
    """
    soft contact model
    """
    def __init__(
            self,
    ):        
        def get_model(cm_params, phi_n, contact_vel):
            """
            returns soft contact model contact vector (frictionless case)
            """
            v_n, v_t = jnp.split(contact_vel, 2)
            # get normal contact
            lam_n = jnp.maximum(
                0.0,-cm_params['K']*phi_n-cm_params['C']*jnp.abs(v_n)
            )
            lam_t = jnp.sign(v_t)*jnp.maximum(
                -cm_params['mu']*lam_n,-cm_params['R']*jnp.abs(v_t)
            )
            return jnp.hstack(
                [
                    lam_n,
                    lam_t
                ]
            )
        
        self.get_model = get_model