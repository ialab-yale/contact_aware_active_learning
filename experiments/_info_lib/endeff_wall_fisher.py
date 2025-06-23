import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad

class Fisher():
    """
    constructs information from contact
    """
    def __init__(self,dyn_model,contact_model):

        # sensor model with no rollouts
        def norollout_contact_sensor_model(cm_params,traj):
            # extract trajectory
            x = traj['x']
            q, qdot = jnp.split(x, 2, axis=1)
            lam = vmap(contact_model.get_model, in_axes=(None,0,0))(
                cm_params,
                vmap(dyn_model.phi_n)(q), 
                vmap(dyn_model.contact_vel)(x)
            )
            return lam
        
        # sensor model with rollouts internally computed
        def contact_sensor_model(cm_params,u,inits):
            traj = dyn_model.rollout(contact_model.get_model,u,inits['x'],cm_params) # inits is a dictionary of ICs
            x = traj['x']

            q, qdot = jnp.split(x, 2, axis=1)
            lam = vmap(contact_model.get_model, in_axes=(None,0,0))(
                cm_params,
                vmap(dyn_model.phi_n)(q), 
                vmap(dyn_model.contact_vel)(x)
            )
            return lam
            return jnp.clip(lam / 100.0, 0.0, 10.0)
        
        self.contact_sensor_model = contact_sensor_model
        self.norollout_contact_sensor_model = norollout_contact_sensor_model

        @jax.jit
        def get_fisher(cm_params,u,inits):
            """
            returns the fisher information
            """
            # gradient of sensor model
            grad_model = jacfwd(contact_sensor_model, argnums=0)(cm_params,u,inits)
            dY = jnp.hstack(
                [
                    jnp.sum(grad_model['mu'], axis=0)
                ]
            )

            # compute outer
            dYdY = vmap(jnp.outer)(dY,dY)

            return jnp.diag(jnp.diagonal(jnp.sum(dYdY,axis=0)))

        @jax.jit
        def get_outer(cm_params,u,inits,parvar):
            """
            T-optimality fim (sum)
            """
            # obtain fisher information
            fish = get_fisher(cm_params,u,inits)

            # return the average trace of the fisher information
            return jnp.trace(parvar*fish)
        
        self.get_outer = get_outer

        def update_paramvar(parvar,cm_params,u,x_init,alpha=1e-6):
            # obtain fisher information
            fish = jnp.clip(alpha*get_fisher(cm_params,u,x_init), 0.0, 0.1)
            return jnp.clip((parvar**-1 + fish)**-1, 1e-10, jnp.inf)
        
        self.update_paramvar = update_paramvar
