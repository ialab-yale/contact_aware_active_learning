import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad, lax
from tqdm import tqdm
import dill as pkl

# local

# predictive sampler
from samplers.predictive_sampler import ContactAwarePredictiveSampler as CAPS

# simulators
from simulators.endeff_block_sim import Sim as ShapeSim
# from simulators.endeff_wall_sim import Sim as FrictionSim

# information classes
from _info_lib.endeff_block_fisher import Fisher as ShapeFisher
from _info_lib.endeff_wall_fisher import Fisher as FrictionFisher

# parameter estimators
from cmap.cmap import ShapeCMAP

class ShapeEstimationExpDesController():
    """
    contact-aware experimental design algorithm
    """
    def __init__(self, dyn_model, contact_model, exp_num_seed):
        
        # instantiate predictive sampler
        pred_sampler = CAPS(
            ctrl_hrzn=25,
            ctrl_bounds=([-1.0,-1.0],[1.0,1.0]),
            pos_bounds=([-1.0,-1.0],[1.0,1.0]),
            vel_bounds=([-0.3,-0.3],[0.3,0.3])
        )

        # logger for storing data
        self.logger = {'x': [], 'xb': [], 'lam': [], 'u': []}

        # initialize estimates dictionary
        self.reslogger = {'l_box': [],'w_box': [],'fim': []}

        # log data
        def log(data):
            self.logger['x'].append(data['x'])
            self.logger['xb'].append(data['xb'])
            self.logger['lam'].append(data['lam'])
            self.logger['u'].append(data['u'])

        # log results
        def log_res(data):
            self.reslogger['l_box'].append(data['l_box'])
            self.reslogger['w_box'].append(data['w_box'])
            self.reslogger['fim'].append(data['fim'])

        # save data
        def save():
            print("saving data")
            with open(f'data/shape_est/exp_experiment/'+'test_trajectory'+'.pkl', 'wb') as file:
                pkl.dump(self.logger, file)
            print("data saved")

        # save results data
        def save_results():
            print("saving results")
            with open(f'data/shape_est/exp_experiment/res/'+'res'+'.pkl', 'wb') as file:
                pkl.dump(self.reslogger, file)
            print("data saved")

        self.log = log
        self.log_res = log_res
        self.save = save
        self.save_results = save_results

 
        # make contact-aware information object
        fish = ShapeFisher(dyn_model,contact_model)
        simulator = ShapeSim(dyn_model,contact_model)
        cmap_solver = ShapeCMAP(dyn_model,contact_model)

        # cost function
        @jax.jit
        def inf_cost(x_init: dict, u: jnp.ndarray, box_params: dict, parvar: jnp.ndarray) -> float:
            """
            information + distance cost jointly optimized
            """
            # rollout trajectories to compute sdf cost
            traj = dyn_model.rollout(contact_model.get_model,u,x_init['x'],x_init['xb'],box_params)
            x = traj['x']
            xb = traj['xb']
            q, qdot = jnp.split(x, 2, axis=1)
            qb, qbdot = jnp.split(xb, 2, axis=1)

            return -1e3*fish.get_outer(box_params,u,x_init,parvar) \
                    + 1e-5*jnp.sum(vmap(dyn_model.phi_n, in_axes=(0,0,None))(q,qb,box_params)**2) \
                    + 1e-3*jnp.sum(qb**2) + 1e-2*jnp.sum(qbdot**2)
        
        self.get_inf_cost = inf_cost

        def run(box_params,T=500):
            """
            runs the experimental design algo
            """
            # initialize nominal plan and control sampling keys
            nominal_plan = pred_sampler.get_init_plan()
            keys = pred_sampler.get_init_keys(exp_num_seed)
            x_init=dyn_model.x
            xblock_init=dyn_model.xb
            parvar = jnp.hstack([10.0,10.0])

            # initialize data
            simulator_data = {
                'x': x_init,
                'xb': xblock_init,
                'lam': jnp.zeros(1),
                'u': jnp.zeros((int(x_init.shape[0]/2),))
            }

            # initialize results data
            results_data = {
                'l_box': box_params['l_box'],
                'w_box': box_params['w_box'],
                'fim': 0.0,
            }

            for t in tqdm(range(T)):
                # dump state-action data into a pkl file
                self.log(simulator_data)
                self.log_res(results_data)

                # get costs and best action from sampled controls
                ustar, star, sample_nomplans, cost_star = pred_sampler.get_action(inf_cost, keys, nominal_plan, simulator_data, box_params, parvar)
                
                # apply control to robot - real params
                simulator_data = simulator.step(ustar)

                # update the nominal plan and keys
                nominal_plan = pred_sampler.update_plan(star,sample_nomplans,keys)
                keys = pred_sampler.update_keys(star,keys)

                # get new estimates of the parameters
                box_params, fim, parvar = cmap_solver.solve(
                    simulator_data, parvar, box_params
                )

                # update results dictionary
                results_data = {
                    'l_box': box_params['l_box'],
                    'w_box': box_params['w_box'],
                    'fim': fim,
                }
                
                # prints
                print('length:   ', box_params['l_box'])
                print('width:    ', box_params['w_box'])
                print('x:        ', simulator_data['x'])
                print('xb:       ', simulator_data['xb'])
                print('force:    ', simulator_data['lam'])
                print('paramvar: ', parvar)
                print('COST:     ', cost_star)
            
            # save the data and results
            self.save()
            self.save_results()

            simulator.animate_sim(exp_str='exp_test')
            
            return 0
        
        self.run = run


class FrictionEstimationExpDesController():
    """
    contact-aware experimental design algorithm
    """
    def __init__(self, dyn_model, contact_model, cm_params: dict, exp_num_seed: int):

        # instantiate predictive sampler
        pred_sampler = CAPS(
            dt = 0.002,
            ctrl_bounds=([-1.0,-1.0],[1.0,1.0]),
            pos_bounds=([0.65,0.5],[0.71,0.8]),
            vel_bounds=([-2.0,-2.0],[2.0,2.0])
        )

        # logger for storing data
        self.logger = {'x': [], 'lam': [], 'u': []}

        # scaled down contact model (mu is not scaled)
        self.cm_params=cm_params

        # log data
        def log(data):
            self.logger['x'].append(data['x'])
            self.logger['lam'].append(data['lam'])
            self.logger['u'].append(data['u'])


        self.log = log

 
        # make contact-aware information object
        fish = FrictionFisher(dyn_model,contact_model)

       # cost function
        @jax.jit
        def inf_cost(x_init: dict, u: jnp.ndarray, cm_params: dict, parvar: jnp.ndarray) -> float:
            """
            information + distance cost jointly optimized
            """
            # rollout trajectories to compute sdf cost
            traj = dyn_model.rollout(contact_model.get_model,u,x_init['x'],cm_params)
            x = traj['x']
            q, qdot = jnp.split(x, 2, axis=1)

            return -fish.get_outer(cm_params,u,x_init,parvar) + jnp.sum(vmap(dyn_model.phi_n)(q)**2)
        
        self.get_inf_cost = inf_cost

        def run(T=1000):
            """
            runs the experimental design algo
            """
            # initialize nominal plan and control sampling keys
            nominal_plan = pred_sampler.get_init_plan()
            keys = pred_sampler.get_init_keys(exp_num_seed)
            x_init=dyn_model.x                
            parvar = jnp.hstack([10.0])

            # initialize data
            data = {
                'x': x_init,
                'lam': jnp.zeros(2),
                'u': jnp.zeros((int(x_init.shape[0]/2),))
            }

            for t in tqdm(range(T)):
                # dump state-action data into a pkl file
                self.log(data)

                # get costs and best action from sampled controls
                ustar, star, sample_nomplans, cost_star = pred_sampler.get_action(inf_cost, keys, nominal_plan, data, self.cm_params, parvar)

                # get contact force
                lam = contact_model.get_model(
                    self.cm_params,
                    dyn_model.phi_n(data['x'][:2]),
                    dyn_model.contact_vel(data['x'])
                    )
                
                # apply control to robot - real params
                states = dyn_model.f(
                    data['x'], 
                    lam,
                    ustar, 
                )

                # update the nominal plan and keys
                nominal_plan = pred_sampler.update_plan(star,sample_nomplans,keys)
                keys = pred_sampler.update_keys(star,keys)

                # update parameter variance
                parvar = fish.update_paramvar(parvar,self.cm_params,ustar,data)

                # update data dictionary
                data = {
                    'x': states['x'],
                    'lam': lam,
                    'u': ustar
                }

            return jnp.array(self.logger['x'])
        
        self.run = run