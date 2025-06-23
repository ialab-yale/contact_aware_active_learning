#Import all of your modules

import mujoco 
import mujoco_viewer 
import numpy as np 
import jax
from numpy.linalg import inv, pinv 
import mediapy as media
import dill as pkl

from _mj_assets_lib._utils.utilts import rot_err, getSitePose, getSiteVel, getSiteJacobian
from _mj_assets_lib._utils.sim import Sim


class Mj_sim():

    def __init__(self,
                 T,
                 trial_num,
                 measurement_key):
        self.franka_sim = Sim(fname='_mj_assets_lib/xml/scene_torque_actuators.xml')
        self.x_ref = None
        self.t_reset = 500
        self.t_max = self.t_reset + T
        self.prev_ee_pos  = None
        self.log = {'lam': [], 'x': [], 'u': []}
    
        def save():
            print("saving data")
            with open(f'data/friction_est/exp_experiment/simdata{trial_num}.pkl', 'wb') as file:
                pkl.dump(self.log, file)
            print("data saved")
        
        def inject_contact_noise(lam_data):
            lam_data = 0.5*jax.random.normal(key=measurement_key, shape=lam_data.shape) + lam_data
            return lam_data

        self.save = save
        self.inject_contact_noise = inject_contact_noise
    
    def log_contact_data(self, model, data):
        
        if data.ncon > 0:
            ee_geom_name = "attachment_geom" # attachment_geom (commented out in the panda urdf)
            wall_geom_name = "wall" 

            # Get geom IDs
            ee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ee_geom_name)
            wall_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wall_geom_name)

            # Iterate through contacts
            for i in range(data.ncon):

                contact = data.contact[i]

                # print(contact)

                # Check if contact is between the ee and wall
                geom_ids = contact.geom

                if (ee_geom_id in geom_ids and wall_geom_id in geom_ids):

                    # Create array to hold the contact force and torque
                    force = np.zeros(6)  # [f_normal, f_friction1, f_friction2, torque_x, torque_y, torque_z]
                    mujoco.mj_contactForce(model, data, i, force)

                    # Extract contact force vector in the contact frame
                    contact_force = force[:3]  # [normal, friction1, friction2]
                    print(f"Contact force vector (normal + friction): {contact_force}")

                    # append to the contact list
                    self.log['lam'].append(np.array([contact_force[0], contact_force[-1]])) # normal force, friction force in z direction wrt ee frame

                    return
        
        else:

            # Create zeros
            force = np.zeros(6)  # [f_normal, f_friction1, f_friction2, torque_x, torque_y, torque_z]

            # Extract contact force vector in the contact frame
            contact_force = force[:3]  # [normal, friction1, friction2]
            print(f"Contact force vector (normal + friction): {contact_force}")

            # append to the contact list
            self.log['lam'].append(np.array([contact_force[0], contact_force[-1]])) # normal force, friction force in z direction wrt ee frame

            return

        self.log['lam'].append(np.array([0.0, 0.0])) 
    
    def check_contacts(self, model, data):
        for i in range(data.ncon):
            contact = data.contact[i]

            # Get geom IDs
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Map to body IDs
            body1_id = model.geom_bodyid[geom1_id]
            body2_id = model.geom_bodyid[geom2_id]

            # Get body names
            body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            print(f"Contact {i}: {body1_name} <-> {body2_name}")

    def log_ee_data(self, model, data):
        (R_ee, p_ee) = getSitePose(model, data, "attachment_site")

        # numerically obtain velocity
        # if self.prev_ee_pos is not None:
        #     v_ee = (p_ee - self.prev_ee_pos)/0.002
        #     self.prev_ee_pos = p_ee.copy()
        # else:
        #     v_ee = np.zeros_like(p_ee)
        #     self.prev_ee_pos = p_ee.copy()

        # get the ee site jacobian
        jac = getSiteJacobian(model, data, "attachment_site")
        jacp, jacr = np.split(jac, 2, axis=0)

        v_ee = jacp @ data.qvel

        # log
        self.log['x'].append(np.array([p_ee[0], p_ee[-1], v_ee[0], v_ee[-1]]))
    
    def log_ctrl_data(self, model, data):
        # get the ee site jacobian
        jac = getSiteJacobian(model, data, "attachment_site")
        jacp, jacr = np.split(jac, 2, axis=0)

        # compute control
        task_force_linear = jacp @ data.ctrl  # 3D linear force vector
        task_force_angular = jacr @ data.ctrl  # 3D torque vector

        # we only care about the linear forces
        self.log['u'].append(np.array([task_force_linear[0], task_force_linear[-1]]))
        
    def get_inertia(self, model, data):
        for body_id in range(model.nbody):
            # Get the inertia matrix from the model
            inertia = model.body_inertia[body_id]
            # Print out the inertia matrix
            print(f"Inertia for body {body_id}: {inertia}")
        
    def getMassMatrix(self, model, data):
        """
        M(q) in R^{nxn}
        """
        mass_matrix = np.zeros((model.nv, model.nv)) 
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        
        return mass_matrix

    def getBiasTerms(self, model, data):
        """
            b(q, qdot) in R^n
        """
        return data.qfrc_bias  

    def getOSMassMatrix(self, sys_mass_matrix, jac):
        """
            M_x(q) = (J * M^{-1} * J^T)^{-1}
        """
        OSMassMatrix = inv(jac@np.linalg.inv(sys_mass_matrix)@jac.T)

        return OSMassMatrix

    def operational_space_control(self, model, data):
        """
        This builds our PD controller using Inverse Kinematics Dynamics
        """

        #Set PD control params
        kp = 80
        kd = 20

        (R_ee, p_ee) = getSitePose(model, data, "attachment_site") 
        (R_des, p_des) = getSitePose(model, data, "target_site") 
        jac = getSiteJacobian(model, data, "attachment_site")    

        #the initial or current pose and target pose, which will be minimized
        err_p = p_des - p_ee 
        err_rot = rot_err(R_des, R_ee) 
        err = np.concatenate([err_p, err_rot])
        err_dot = jac@data.qvel

        xddot_des = kp * err + kd * (-err_dot)

        mass_matrix = self.getMassMatrix(model, data)
        bias        = self.getBiasTerms(model,data)
        
        op_mass_matrix = self.getOSMassMatrix(mass_matrix, jac)

        tau = jac.T @ op_mass_matrix @ xddot_des + bias
        return tau

    def traj_tracking_controller(self, model,data,t):

        #Set PD control params
        kp = 500 
        kd = 100

        (R_ee, p_ee) = getSitePose(model, data, "attachment_site") 
        p_des = np.hstack([self.x_ref[t, 0], 0.0, self.x_ref[t, 1]]) # 0.05
        (R_des, _) = getSitePose(model, data, "target_site") 

        jac = getSiteJacobian(model, data, "attachment_site")    

        #the initial or current pose and target pose, which will be minimized
        err_p = p_des - p_ee 
        err_rot = rot_err(R_des, R_ee) 
        err = np.concatenate([err_p, err_rot])
        err_dot = jac@data.qvel

        xddot_des = kp * err + kd * (-err_dot)

        mass_matrix = self.getMassMatrix(model, data)
        bias        = self.getBiasTerms(model,data)
        
        op_mass_matrix = self.getOSMassMatrix(mass_matrix, jac)

        tau = jac.T @ op_mass_matrix @ xddot_des + bias
        return tau

    def run(self, x_ref, render=True):
        sim = self.franka_sim
        self.x_ref = x_ref
        t_max = self.t_max
        if len(sim.model.key_qpos) != 0:
            sim.data.qpos = sim.model.key_qpos
        mujoco.mj_step(sim.model, sim.data)
        if render:
            # viewer and frames
            viewer = mujoco_viewer.MujocoViewer(sim.model, sim.data)
            viewer.vopt.frame = 3

            # Enable contact force visualization (equivalent to pressing 'c') 
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

            # Disable shadows (equivalent to pressing 'o')
            viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

        for t in range(t_max):
            print(f'{t}. ', end='')

            # go to init state and follow the trajectory
            if t > self.t_reset: 
                ctrl_method = self.traj_tracking_controller
                u = ctrl_method(sim.model, sim.data,t-self.t_reset)
            else: 
                ctrl_method = self.operational_space_control
                u = ctrl_method(sim.model, sim.data)

            # set control and step
            sim.data.ctrl = u 
            mujoco.mj_step(sim.model, sim.data)

            # log data
            self.log_contact_data(sim.model, sim.data)
            self.log_ee_data(sim.model, sim.data)
            self.log_ctrl_data(sim.model, sim.data)
            
            # render
            if render:
                if viewer.is_alive:
                    viewer.render()
        
        # convert to numpy types and save sim data
        self.log['lam'] = self.inject_contact_noise(np.array(self.log['lam'])) # inject artificial noise to the contact data
        self.log['x'] = np.array(self.log['x'])
        self.log['u'] = np.array(self.log['u'])
        self.save()

        # close
        if render:
            viewer.close()
        
        return self.log