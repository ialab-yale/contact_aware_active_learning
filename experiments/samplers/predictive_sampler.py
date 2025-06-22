import jax
import jax.numpy as jnp
from jax import vmap

class ContactAwarePredictiveSampler:
    """
    predictive sampling functions
    """
    def __init__(
            self,
            dt:float = 0.01,
            ctrl_hrzn:int = 10,
            ctrl_bounds:tuple = ([-1.0],[1.0]),
            pos_bounds:tuple = ([-1.0], [1.0]),
            vel_bounds:tuple = ([-0.5], [0.5]),
            num_splines: float = 10

    ):
        # set default control limits
        u_min_default, u_max_default = jnp.hstack(ctrl_bounds[0]), jnp.hstack(ctrl_bounds[1])

        # set kinematic limits
        q_min, q_max = jnp.hstack(pos_bounds[0]), jnp.hstack(pos_bounds[1])
        qdot_min, qdot_max = jnp.hstack(vel_bounds[0]), jnp.hstack(vel_bounds[1])

        # sampling variance (keep at 1)
        var = 1.0

        # control horizon
        self.P = ctrl_hrzn

        def get_ctrl_limits(q: jnp.ndarray, qdot: jnp.ndarray) -> tuple:
            """
            returns new control limits given current system state
            """
            u_max = jnp.minimum(
                jnp.minimum(
                    u_max_default, (qdot_max - qdot)/dt
                ), 
                (q_max - q)/dt
            )
            u_min = jnp.maximum(
                jnp.maximum(
                    u_min_default, (qdot_min - qdot)/dt
                ), 
                (q_min - q)/dt
            )
            return u_min, u_max
        
        def get_init_nominal_plan(): # TODO
            """
            initializes the nominal plan
            """
            if len(ctrl_bounds[0])==1:
                return jnp.zeros((self.P,))
            else:
                return jnp.zeros((self.P,len(ctrl_bounds)))
        
        def get_init_keys(seed):
            """
            initializes control sampling keys
            """
            return jax.random.split(jax.random.PRNGKey(seed),num=num_splines)
        
        self.get_init_plan = get_init_nominal_plan
        self.get_init_keys = get_init_keys

        def update_plan(star, sample_nomplan, keys):
            # update nominal_plan
            return sample_nomplan[star] \
                + var**2*jax.random.normal(keys[star], shape=sample_nomplan[star].shape)
            
        def update_keys(star, keys):
            # update control sampling keys
            return jax.random.split(keys[star],num=num_splines)
        
        self.update_plan = update_plan
        self.update_keys = update_keys
        
        def cubic_hermite_spline(key, nominal_plan, x):
            """
            returns the control splines across sampled from plan
            """
            # obtain nominal plan samples
            u_min, u_max = get_ctrl_limits(*jnp.split(x, 2))
            sample_nomplan = jnp.clip((var**2 * jax.random.normal(key, shape=nominal_plan.shape) + nominal_plan), u_min,u_max)

            # obtain weights
            w = jnp.linspace(0.0, 1.0, self.P)
            w_vec = jnp.vstack([jnp.ones_like(w), w, w**2, w**3]).T

            def compute_ctrl(w_vec_T_j, theta_jm1, theta_j, theta_jp1, theta_jp2):
                """
                computes control spline across all samples
                """
                # compute phis via finite difference method
                phi_j = 0.5 * ((theta_jp1 - theta_j)/dt + (theta_j - theta_jm1)/dt)
                phi_jp1 = 0.5 * ((theta_jp2 - theta_jp1)/dt + (theta_jp1 - theta_j)/dt)
                theta_vec = jnp.vstack([theta_j, phi_j, theta_jp1, phi_jp1])

                # scaling matrix
                scaling_mat = jnp.array([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, dt, 0.0, 0.0],
                                        [-3.0, -2.0*dt, 3.0, -dt],
                                        [2.0, dt, -2.0, dt]])

                return w_vec_T_j @ scaling_mat @ theta_vec
            
            return jnp.clip(
                    vmap(
                    compute_ctrl, 
                    in_axes=(0,0,0,0,0)
                )(
                    w_vec[1:-2], 
                    sample_nomplan[:-3], 
                    sample_nomplan[1:-2], 
                    sample_nomplan[2:-1], 
                    sample_nomplan[3:]
                ),
                u_min_default,
                u_max_default
            ), sample_nomplan

        def get_best_act(cost_fun,keys,nominal_plan,x_init,physics_params,parvar):
            """
            returns best control input based on cost
            """
            # get control splines
            control_splines, sample_nomplans = vmap(
                cubic_hermite_spline, in_axes=(0,None,None)
            )(keys,nominal_plan,x_init['x'])

            # use splines to compute costs
            costs = vmap(cost_fun, in_axes=(None,0,None,None))(x_init,control_splines,physics_params,parvar)
            star = jnp.argmin(costs)

            return control_splines[star][0], star, sample_nomplans, costs[star]
        
        self.get_action = get_best_act
        self.spline = cubic_hermite_spline