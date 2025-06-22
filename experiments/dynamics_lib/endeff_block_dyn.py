import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, lax

class EndEffBlockDyn():
    """
    constructs system dynamics and model params
    """
    def __init__(
            self,
            x_init: jnp.ndarray,
            xblock_init: jnp.ndarray,
            mass: float,
            radius: float,
            dt = 0.01 
    ):
        super().__init__()
        # model params
        self.nbox = int(xblock_init.shape[0]/2)
        self.n = int(x_init.shape[0]/2)
        self._x = x_init
        self.x = x_init
        self._xb = xblock_init
        self.xb = xblock_init
        self.M = mass*jnp.eye(self.n)
        self.r = radius
        self.g = 9.81

        # box params
        # box_params = {
        #     'l_box': self.l_box,
        #     'w_box': self.w_box
        # }

        # sdf
        def phi_n(q: jnp.ndarray, qb: jnp.ndarray, box_params: dict) -> jnp.ndarray:
            """
            returns the signed distance function for point mass body with a box
            Args:
              `q`[jnp.ndarray]: pose of ee
              `qb`[jnp.ndarray]: pose of block
            """
            def softmax(a, b, eps=0.001):
                """
                softmax activation
                """
                return 0.5 * (a + b + jnp.sqrt((a-b)**2 + eps**2))
            def R(angle: float):
                """
                z-Rotation matrix
                """
                return jnp.array([[jnp.cos(angle), -jnp.sin(angle)],
                                [jnp.sin(angle), jnp.cos(angle)]]).reshape((2,2))
            # split up the state vector of box
            qbx, qby, qbphi = jnp.split(qb, self.nbox)
            # sdf vector
            phix, phiy = jnp.split(
                jnp.abs(
                R(-qbphi)@(q - jnp.hstack([qbx, qby]))
                ) - jnp.hstack(
                    [
                        0.5*box_params['l_box'] + self.r,
                        0.5*box_params['w_box'] + self.r
                    ]
                ),
                2
            )
            return (softmax(phix,0.0)**2 + softmax(phiy,0.0)**2)**0.5 \
                    -softmax(-softmax(phix,phiy),0.0)
        
        self.phi_n = phi_n

        # contact jacobians
        def J(q: jnp.ndarray, qb: jnp.ndarray, box_params: dict) -> jnp.ndarray:
            return jnp.hstack(jacfwd(phi_n,argnums=0)(q,qb,box_params))
        
        def Jblock(q: jnp.ndarray, qb: jnp.ndarray, box_params: dict) -> jnp.ndarray:
            return jnp.hstack(jacfwd(phi_n,argnums=1)(q,qb,box_params))
        
        self.J = J
        self.Jblock = Jblock

        # contact velocity
        def Jqdot(x,xb,box_params):
            q, qdot = jnp.split(x, 2)
            qb, qdotb = jnp.split(xb, 2)
            return J(q,qb,box_params)*qdot
        
        self.contact_vel = Jqdot

        # system dynamics
        def f(x: jnp.ndarray, xb: jnp.ndarray, lam: jnp.ndarray, u: jnp.ndarray, box_params: dict) -> jnp.ndarray:
            """
            using semi-implicit euler, no gravity
            """
            q, qdot = jnp.split(x, 2)
            qb, qbdot = jnp.split(xb,2)
            qdot_next = qdot + dt*(jnp.linalg.inv(self.M)@(J(q,qb,box_params).T*lam + u))
            qbdot_next = qbdot + dt*(jnp.linalg.inv(box_params['Mblock'])@(Jblock(q,qb,box_params).T*lam)
                        - box_params['ground_fric']*self.g*jnp.hstack([jnp.sign(qbdot)]))
            return {
                'x': jnp.hstack(
                    [
                        q + dt*qdot_next,
                        qdot_next
                    ]
                ),
                'xb': jnp.hstack(
                    [
                        qb + dt*qbdot_next,
                        qbdot_next
                    ]
                )
            }
        
        self.f = f

        # rollout
        def rollout(contact_model, u:jnp.ndarray, _x_init: jnp.ndarray, _xblock_init: jnp.ndarray, box_params: dict) -> tuple:
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
                    _x['xb'], 
                    contact_model(
                        phi_n(_x['x'][:self.n],_x['xb'][:self.nbox],box_params),
                        Jqdot(_x['x'],_x['xb'],box_params),
                    ),
                    _u,
                    box_params
                ), _x
            return lax.scan(
                f=step,
                init={'x':_x_init,'xb':_xblock_init},
                xs=u,
                length=u.shape[0]
            )[-1]
        
        # rollout just the box dynamics given ee data
        def partialrollout(contact_model, u:jnp.ndarray, _x_traj: jnp.ndarray, _xblock_init: jnp.ndarray, box_params: dict) -> tuple:
            """
            returns a full trajectory
            """
            @jax.jit
            def step(_x_b, _in):
                """
                steps forward the dynamical model
                """
                x_t = _in['x']
                _u = _in['u']
                return f(
                    x_t, 
                    _x_b, 
                    contact_model(
                        phi_n(x_t[:self.n],_x_b[:self.nbox],box_params),
                        Jqdot(x_t,_x_b,box_params),
                    ),
                    _u,
                    box_params
                )['xb'], _x_b
            return lax.scan(
                f=step,
                init=_xblock_init,
                xs={'u': u, 'x': _x_traj},
                length=u.shape[0]
            )[-1]
        
        self.rollout = rollout
        self.partialrollout = partialrollout
    
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
    
    # block
    @property
    def xb(self):
        return self._xb
    
    @xb.setter
    def xb(self, _xb):
        if _xb.shape == self._xb.shape:
            self._xb = _xb
        else:
            print(f'got {_xb.shape}, should be {self._xb.shape}')
            raise ValueError("Wrong dimension for state")
        
    @xb.getter
    def xb(self):
        return self._xb