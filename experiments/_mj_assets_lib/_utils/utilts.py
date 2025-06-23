import numpy as np
import scipy as sp
import mujoco

def hat(w):
    return np.array([
        [0., -w[2], w[1]],
        [w[2], 0., -w[0]],
        [-w[1], w[0], 0.]])

def unhat(R):
    return np.array([R[2,1], R[0,2], R[1,0]])

def rot_err(R1, R2):
    return unhat(sp.linalg.logm(R1@R2.T))


def getSitePose(model, data, site_name):
    """
        This function takes in the robot model, data, and the site name of the reference frame. 
        The output of the function returns a tuple of a rotation matric R and a position p 
        relative to the world frame. 
    """
    # mujoco specific function
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    # data structure for obtaining the position and orientation of a site reference frame
    # note that rotation matrices are formated as a 9-dim vector and need to be reshaped
    p = data.site_xpos[site_id]
    R = data.site_xmat[site_id].reshape((3,3))
    return (R, p)

def getSiteVel(model, data, site_name):
    """
        This function takes in the robot model, data, and the site name of the reference frame. 
        The output of the function returns a tuple of a rotation matric R and a position p 
        relative to the world frame. 
    """
    # mujoco specific function
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    # data structure for obtaining the position and orientation of a site reference frame
    # note that rotation matrices are formated as a 9-dim vector and need to be reshaped
    pos_vel = np.array(data.site_xvelp[site_id])
    ori_vel = np.array(data.site_xvelr[site_id])
    return (ori_vel, pos_vel)

def getSiteJacobian(model, data, site_name):
    """
        This function takes in the robot model, data, and the site name of the reference frame. 
        The output is the total jacobian matrix. Note that mujoco gives jacobian 
        position and rotation matrices which need to be stacked together.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, body_id)
    return np.concatenate([jacp, jacr], axis=0)