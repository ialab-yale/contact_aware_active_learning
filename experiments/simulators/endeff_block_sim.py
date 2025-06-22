import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad, lax
import dill as pkl
from tqdm import tqdm

# local
from obj_models_lib.box import Box
from mpl_vis_lib.endeff_block_ani import animate

class Sim():
    """
    simulation
    """
    def __init__(
            self,
            dyn_model,
            contact_model,
            measurement_key=jax.random.PRNGKey(0),
    ):  
        
        # initialize the key
        self.measurement_key = measurement_key

        # initialize trajectory lists
        self.traj = {'x': [], 'xb': []}
        self.prev_x = {'x': dyn_model.x, 'xb': dyn_model.xb}

        # instantiate box object with true inaccessible params
        box_params=Box(
                l_box=0.5, 
                w_box=0.3, 
                massblock=0.25,
                inertiablock=0.004,
                ground_fric=0.2 
            ).get_dict()

        # save real box parameters
        with open('data/shape_est/'+'box_params'+'.pkl', 'wb') as file:
            pkl.dump(box_params, file)
        
        def step(_u):
                """
                steps forward the dynamical model
                """
                lam = contact_model.get_model(
                        dyn_model.phi_n(self.prev_x['x'][:dyn_model.n],self.prev_x['xb'][:dyn_model.nbox],box_params),
                        dyn_model.contact_vel(self.prev_x['x'],self.prev_x['xb'],box_params),
                    )
                x_dict = dyn_model.f(
                    self.prev_x['x'], 
                    self.prev_x['xb'], 
                    lam,
                    _u,
                    box_params
                )

                # add trajectories to a list
                self.traj['x'].append(x_dict['x'])
                self.traj['xb'].append(x_dict['xb'])

                # update previous state
                self.prev_x = x_dict

                # add artificial noise for measurements
                measured_lam, measured_xb, measured_x = get_measurements(lam, x_dict['xb'], x_dict['x'])

                return {
                    'x': measured_x,
                    'xb': measured_xb,
                    'u': _u,
                    'lam': measured_lam
                }
        
        self.step = step
        
        def get_measurements(lam_data, xblock_arr, x_arr):
            _, self.measurement_key = jax.random.split(self.measurement_key)
            lam_data = 0.0*jax.random.normal(key=self.measurement_key, shape=lam_data.shape) + lam_data
            xblock_arr = 0.0*jax.random.normal(key=self.measurement_key, shape=xblock_arr.shape) + xblock_arr
            x_arr = 0.0*jax.random.normal(key=self.measurement_key, shape=x_arr.shape) + x_arr
            return lam_data, xblock_arr, x_arr

        def animate_simulation(exp_str='exptest'):
             self.traj['x'], self.traj['xb'] = np.array(self.traj['x']), np.array(self.traj['xb'])
             animate(self.traj, 
                    box_params, 
                    filename=exp_str)

        self.animate_sim = animate_simulation