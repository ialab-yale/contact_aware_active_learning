import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, grad, lax
import dill as pkl

# locals
from info_lib.endeff_block_fisher import Fisher as ShapeFisher
from info_lib.endeff_wall_fisher import Fisher as FrictionFisher

from solvers.augmented_lagrangian import AugmentedLagrangeSolver as ALS

class ShapeCMAP():

    def __init__(self,dyn_model,contact_model):

        # instantiate the solver vars
        self.solver: ALS = None
        
        #instantiate the fisher information object
        Cofish = ShapeFisher(dyn_model,contact_model)

        # get box param predicted
        def get_predicted_box_params(params,args):
            return {
                        'l_box': params['l_box'],
                        'w_box': params['w_box'],
                        'Mblock': args['box_params']['Mblock'],
                        'ground_fric': args['box_params']['ground_fric']
                    }

        # solve the optimization
        def solve(sim_data_dict, paramvariance, box_params):
            """
            estimates parameters via maximum a-priori likelihood estimation
            """
            # optimization dictionaires
            args = {
                'sim_data': sim_data_dict,
                'box_params': box_params,
            }
            # params = box_params
            params = {
                'l_box': box_params['l_box'],
                'w_box': box_params['w_box']
            }

            # cost
            def loss(params,args):
                # extract simulation data
                x = jnp.array(args['sim_data']['x'])
                xb = jnp.array(args['sim_data']['xb'])
                lam_sim = jnp.array(args['sim_data']['lam'])

                lam_pred = contact_model.get_model(
                    dyn_model.phi_n(x[:dyn_model.n],xb[:dyn_model.nbox],get_predicted_box_params(params,args)),
                    dyn_model.contact_vel(x,xb,get_predicted_box_params(params,args))
                )

                # return the loss = contact least squares discrepancy + parameter least squares discrepancy
                return jnp.sum((lam_sim - lam_pred)**2) \
                    + paramvariance[0]**-1 * jnp.sum((params['l_box'] - args['box_params']['l_box'])**2) \
                    + paramvariance[1]**-1 * jnp.sum((params['w_box'] - args['box_params']['w_box'])**2)
            
            # constraints
            def ineq_constr(params,args):
                return jnp.hstack(
                    [
                        # parameter feasibility
                        -params['l_box'],
                        -params['w_box']
                    ]
                )
            
            # solve
            if self.solver is None:
                self.solver = ALS(
                    x0=params,
                    loss=loss,
                    eq_constr=lambda params,args : jnp.array(0.0),
                    ineq_constr=ineq_constr,
                    args=args,
                    step_size=0.01 
                )
            else:
                self.solver.reset_solver(
                    x0=params,
                    loss=loss,
                    eq_constr=lambda params,args : jnp.array(0.0),
                    ineq_constr=ineq_constr,
                    args=args,
                )

            self.solver.solve(max_iter=500, alpha=1.0001)
            sol = self.solver.get_solution()

            # return sol
            return get_predicted_box_params(sol,args), Cofish.get_outer(
                get_predicted_box_params(sol,args),
                jnp.array(args['sim_data']['u']),
                {'x': jnp.array(args['sim_data']['x']), 'xb': jnp.array(args['sim_data']['xb'])},
                paramvariance
            ), Cofish.update_paramvar(
                paramvariance,
                get_predicted_box_params(sol,args),
                jnp.array(args['sim_data']['u']),
                {'x': jnp.array(args['sim_data']['x']), 'xb': jnp.array(args['sim_data']['xb'])},
            )
        
        self.solve = solve

class FrictionCMAP():

    def __init__(self,dyn_model,contact_model,cm_params):
        
        #instantiate the fisher information object
        fish = FrictionFisher(dyn_model,contact_model)
        
        # get sensor model predictions
        def get_contact_prediction(cm_params, traj):
            """
            returns the contact vectors for all time given the prediction model
            """
            # extract trajectory
            x = traj['x']
            q, qdot = jnp.split(x, 2, axis=1)
            return vmap(contact_model.get_model, in_axes=(None,0,0))(
                cm_params,
                vmap(dyn_model.phi_n)(q), 
                vmap(dyn_model.contact_vel)(x)
            )
        
        def get_friction_prediction(data, cm_params):

            def get_lam_t(lam_n, contact_vel):
                v_n, v_t = jnp.split(contact_vel, 2)
                eps = 0.001
                mj_limpref_scalar = 0.15 # 0.25, 0.125
                return -mj_limpref_scalar*cm_params['mu']*lam_n*v_t/(jnp.sqrt(jnp.abs(v_t)**2 + eps**2))
                return jnp.sign(v_t)*jnp.maximum(
                            -cm_params['mu']*lam_n,-cm_params['R']*jnp.abs(v_t)
                        )

            lam_predicted = vmap(get_lam_t)(
                data['lam'][:,0],
                vmap(dyn_model.contact_vel)(data['ee_pos']),
            )
            return lam_predicted

        self.get_contact_prediction = get_contact_prediction
        self.get_friction_prediction = get_friction_prediction

        def get_est_cm_params(params, args):
            cm_params = {
                'K': args['cm_params']['K'],
                'C': args['cm_params']['C'],
                'mu': params['mu'],
                'R': args['cm_params']['R']
            }
            return cm_params
        
        # solve the optimization
        def solve(sim_data_dict, paramvariance, trial_num):
            """
            estimates parameters via maximum a-priori likelihood estimation
            lam_data, u_sim, traj
            """
            # optimization dictionaires
            args = {
                'sim_data': sim_data_dict,
                'cm_params': cm_params,
            }
            params = {'mu': cm_params['mu']}

            # cost
            def loss(params,args):
                def loss_experiment(experiment_data):
                    # ee_data = experiment_data['ee_pos']
                    # u = experiment_data['u']
                    lam_data = experiment_data['lam']
                    lam_t_prediction = get_friction_prediction(
                                experiment_data,
                                get_est_cm_params(params,args),
                            )
                    # return jnp.sum(lam)
                    return jnp.mean(
                        (lam_t_prediction - lam_data[:,-1])**2
                    )  
                total_loss = 0.0
                for key in range(trial_num+1):
                    total_loss += loss_experiment(args['sim_data'][f'{key}'])
                return total_loss + paramvariance[0]**-1 * (params['mu'] - args['cm_params']['mu'])**2 \
            
            # constraints
            def ineq_constr(params,args):
                return jnp.hstack(
                    [
                        # parameter feasibility
                        -params['mu'],
                        params['mu'] - 1.0,
                    ]
                )
            
            # solve
            opt = ALS(
                x0=params,
                loss=loss,
                eq_constr=lambda params,args : jnp.array(0.0),
                ineq_constr=ineq_constr,
                args=args,
                step_size=0.01 # 0.01 
            )
            opt.solve(max_iter=100, alpha=1.0001) # 5000
            sol = opt.get_solution()

            print(loss(sol,args))

            # return sol
            return sol, fish.get_outer(
                get_est_cm_params(sol,args),
                args['sim_data'][f'{trial_num}']['u'],
                {'x': dyn_model.x},
                paramvariance
            ), fish.update_paramvar(
                paramvariance,
                get_est_cm_params(sol,args),
                args['sim_data'][f'{trial_num}']['u'],
                {'x': dyn_model.x},
            )
        
        self.solve = solve