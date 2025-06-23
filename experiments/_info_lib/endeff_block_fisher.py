import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad

class Fisher():
    """
    constructs information from contact
    """
    def __init__(self,dyn_model,contact_model):
        
        # sensor model with rollouts internally computed
        @jax.jit
        def contact_sensor_model(box_params,u,x_init):
            traj = dyn_model.rollout(contact_model.get_model,u,x_init['x'],x_init['xb'],box_params)
            x = traj['x']
            xb = traj['xb']

            q, qdot = jnp.split(x, 2, axis=1)
            qb, qbdot = jnp.split(xb, 2, axis=1)
            lam = vmap(contact_model.get_model)(
                vmap(dyn_model.phi_n, in_axes=(0,0,None))(q,qb,box_params), 
                vmap(dyn_model.contact_vel, in_axes=(0,0,None))(x,xb,box_params)
            )

            return lam
            return jnp.clip(lam / 100.0, 0.0, 10.0)
        
        self.contact_sensor_model = contact_sensor_model

        @jax.jit
        def get_fisher(params,u,x_init):
            """
            returns the fisher information
            """
            # gradient of sensor model
            grad_model = jacfwd(contact_sensor_model, argnums=0)(params,u,x_init)
            dY = jnp.hstack(
                [
                    grad_model['l_box'],
                    grad_model['w_box']
                ]
            )

            # compute outer
            dYdY = vmap(jnp.outer)(dY,dY)

            return jnp.diag(jnp.diagonal(jnp.sum(dYdY,axis=0)))

        @jax.jit
        def get_outer(params,u,x_init,parvar):
            """
            T-optimality fim (sum)
            """
            # obtain fisher information
            fish = get_fisher(params,u,x_init)

            # return the average trace of the fisher information
            return jnp.trace(jnp.diag(parvar)@fish)
        
        self.get_outer = get_outer

        @jax.jit
        def update_paramvar(parvar,params,u,x_init,alpha=1e-3):
            # obtain fisher information
            fish = jnp.clip(alpha*get_fisher(params,u,x_init), 0.0, jnp.inf)
            return jnp.clip((parvar**-1 + jnp.diagonal(fish))**-1, 1e-10, jnp.inf)
        
        self.update_paramvar = update_paramvar
