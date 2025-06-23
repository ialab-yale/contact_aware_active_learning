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
            model_params: dict,
    ):
        # model parameters
        K = model_params['K']
        C = model_params['C']
        
        def get_model(phi_n, contact_vel):
            """
            returns soft contact model contact vector (frictionless case)
            """
            # split up the contact vel vector
            v_n,v_t = jnp.split(contact_vel, 2)
            # get normal contact
            lam_n = jnp.maximum(
                0.0,-K*phi_n-C*v_n
            )
            return jnp.hstack(
                [
                    lam_n
                ]
            )
        
        self.get_model = get_model
