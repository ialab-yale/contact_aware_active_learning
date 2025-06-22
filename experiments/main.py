import os
import time
import jax.numpy as jnp
import jax
import dill as pkl
import click

# local imports
from obj_models_lib.box import Box
from contact_models_lib.contact_model_1D import SoftContactModel as SCM_1D
from contact_models_lib.contact_model_2D import SoftContactModel as SCM_2D
from dynamics_lib.endeff_block_dyn import EndEffBlockDyn
from controllers.controllers import ShapeEstimationExpDesController, FrictionEstimationExpDesController
from dynamics_lib.endeff_wall_dyn import EndEffWallDyn
from simulators.endeff_wall_sim import Mj_sim
from cmap.cmap import FrictionCMAP

@click.command()
@click.option('--exp_type', type=click.Choice(['shape', 'friction']),
              default='shape', help='Which experiment to run.')
@click.option('--num_trials', default=10,
              help='Number of experimental trials.')
@click.option('--total_time_iter', default=1000,
              help='Number of time iterations for each experiment.')
@click.option('--measure_seed', default=10,
              help='measurement seed used to inject artifical noise in contact measurements.')

class Main():

    def __init__(self, exp_type: str, num_trials: float, total_time_iter: int, measure_seed: int):
        assert exp_type == 'shape' or exp_type == 'friction'
        assert num_trials > 0
        assert total_time_iter > 0
        assert measure_seed >= 0

        def run_shape_experiment():

            # Create the folder (and parent directories if they don’t exist)
            os.makedirs(f'data/shape_est/exp_experiment/res', exist_ok=True)

            # instantiate the expdesign class
            expdes = ShapeEstimationExpDesController(
                dyn_model=EndEffBlockDyn(
                    x_init=jnp.array([0.0,-0.4,0.0,0.0]), 
                    xblock_init=jnp.array([0.0,0.0,0.0,0.0,0.0,0.0]),
                    mass=1.0,
                    radius=0.025,
                ),
                contact_model=SCM_1D(
                    model_params={'K': 100.0, 'C': 0.0} # 500, 0.0
                ),
                exp_num_seed=measure_seed
            )                     

            # run
            expdes.run(
                        box_params=Box(
                            l_box=1.0,      
                            w_box=1.0,      
                            massblock=0.25,
                            inertiablock=0.004,
                            ground_fric=0.2     
                        ).get_dict(),
                        T=total_time_iter
                    )
        
        def run_friction_experiments():

            cm_params = {'K': 700000.0, 'C': 20000.0, 'mu': 0.9, 'R': 4000.0}
            init_cm_params = cm_params
            fim = 0.0
            paramvariance = jnp.array([[10.0]])
            res_dict = {'mu': [], 'fim': []}
            start_time = time.time()
            sim_data_dict = {} 
            key = jax.random.PRNGKey(measure_seed) 

            # reinstantiate the robot to reset the ee state
            def robot_obj():
                return EndEffWallDyn(
                        x_init=jnp.array([0.65,0.8,0.0,0.0]), 
                        ee_radius=0.05, # 0.01 
                        ee_mass=0.05 # 0.5
                    )
            
            # reinstantiate the contact model every experiment
            def contact_model_obj():
                return SCM_2D()
                        
            def split_measurement_key(key):
                return jax.random.split(key)[-1]
            
            print(f'prior: {cm_params}')
                    
            # Create the folder (and parent directories if they don’t exist)
            os.makedirs(f'data/friction_est/exp_experiment/res', exist_ok=True)
            
            # begin experiments over trials
            for trial in range(num_trials): # number of trials

                # append results to list
                res_dict['mu'].append(cm_params['mu'])
                res_dict['fim'].append(fim)

                # experimental design
                xref = FrictionEstimationExpDesController(robot_obj(),contact_model_obj(),cm_params['mu'],measure_seed).run(total_time_iter)
                
                # simulate the reference trajectory
                data = Mj_sim(T=total_time_iter,
                            trial_num=trial,
                            measurement_key=key).run(xref)

                # split key
                key = split_measurement_key(key)

                # update the sim data dictionary
                sim_data_dict.update(
                    {f'{trial}': {
                        'lam': data['lam'],
                        'ee_pos': data['x'],
                        'u': data['u']
                    }}
                )

                # estimate parameters via cmap
                new_parestimates, fim, paramvariance = FrictionCMAP(
                    robot_obj(),contact_model_obj(),cm_params
                ).solve(sim_data_dict, paramvariance[0],trial)

                # prints
                print('trial: ', trial)
                print('new estimates: ', new_parestimates)
                print('fim: ', fim)
                print('paramvariance: ', paramvariance)

                # reinstantiate the box
                cm_params.update({'mu': new_parestimates['mu']})
                print(cm_params)
            
            print('elapsed time: ', (time.time() - start_time) * (1/60), ' minutes')

            # save data
            print("saving data")
            with open(f'data/exp_experiment/res/results_with_noise{str(init_cm_params['mu']).replace('.', '')}.pkl', 'wb') as file:
                pkl.dump(res_dict, file)
            print("data saved")

        if exp_type == 'shape': run_shape_experiment()
        elif exp_type == 'friction': run_friction_experiments()
            
Main()