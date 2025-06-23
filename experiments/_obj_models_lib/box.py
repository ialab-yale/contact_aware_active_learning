import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad

class Box:
    """
    contructs box parameters
    """
    def __init__(
            self,
            l_box: float,
            w_box: float,
            massblock: float,
            inertiablock: float,
            ground_fric: float,
    ):

        def get_dict():
            return {
                'l_box': l_box,
                'w_box': w_box,
                'Mblock': (massblock*jnp.eye(3)).at[-1,-1].set(inertiablock),
                'ground_fric': ground_fric,
            }
        
        self.get_dict = get_dict