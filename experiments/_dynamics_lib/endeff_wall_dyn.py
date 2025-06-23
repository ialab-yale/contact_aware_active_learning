import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, lax

class EndEffWallDyn():
    """
    constructs system dynamics and model params
    """
    def __init__(
            self,
            x_init: jnp.ndarray,
            ee_radius: float,
            ee_mass: float,
            dt = 0.002
    ):
        super().__init__()
        # model params
        self.n = int(x_init.shape[0]/2)
        self._x = x_init
        self.x = x_init
        self.r_ee = ee_radius
        self.wall_xpos = 0.7 #0.685

        # sdf
        def phi_n(q: jnp.ndarray) -> jnp.ndarray:
            """
            returns the signed distance function for point mass body with a box
            Args:
              `q`[jnp.ndarray]: pose of ee right finger (left finger is mirrored from right)
              `qb`[jnp.ndarray]: pose of block
            """
            qx, qy = jnp.split(q, 2)
            return self.wall_xpos - qx - self.r_ee
        
        self.phi_n = phi_n

        # contact jacobians
        def J() -> jnp.ndarray:
            return jnp.array([[-1, 0],
                           [ 0, 1]])
        
        self.J = J

        # contact velocity
        def Jqdot(x):
            q, qdot = jnp.split(x, 2)
            return J()@qdot
        
        self.contact_vel = Jqdot

        # system dynamics
        def f(x: jnp.ndarray, lam: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            """
            using semi-implicit euler, no gravity
            `lam` is considered to be the contact force by the right finger, but assumed identical to the left finger contact
            """
            q, qdot = jnp.split(x, 2)
            qdot_next = qdot + dt*(J()@lam + u)            # assume ee mas of 1
            return {
                'x': jnp.hstack(
                    [
                        q + dt*qdot_next,
                        qdot_next
                    ]
                )
            }
        
        self.f = f

        # rollout
        def rollout(
                contact_model, 
                u:jnp.ndarray, 
                _x_init: jnp.ndarray, 
                cm_params: dict
        ) -> tuple:
            """
            returns a full trajectory
            """
            @jax.jit
            def step(_x, _u):
                """
                steps forward the dynamical model
                """
                return f(
                    _x['x'], 
                    contact_model(
                        cm_params,
                        phi_n(_x['x'][:self.n]),
                        Jqdot(_x['x']),
                    ),
                    _u,
                ), _x
            return lax.scan(
                f=step,
                init={'x':_x_init},
                xs=u,
                length=u.shape[0]
            )[-1]
        
        self.rollout = rollout
    
    # ee
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, _x):
        if _x.shape == self._x.shape:
            self._x = _x
        else:
            print(f'got {_x.shape}, should be {self._x.shape}')
            raise ValueError("Wrong dimension for state")
        
    @x.getter
    def x(self):
        return self._x