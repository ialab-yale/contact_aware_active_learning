import mujoco 
import numpy as np

class Sim(object):
    """
    Simple object that acts as a wrapper for the 
    Mujoco model and data classes. 
    """
    def __init__(self, fname, rand_rgba=False) -> None:
        self.model   = mujoco.MjModel.from_xml_path(fname)
        self.data    = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        if rand_rgba:
            for i in range(self.model.geom_rgba.shape[0]):
                rand_color = list(np.random.choice(range(256), size=3)/256)
                self.model.geom_rgba[i,:3] = rand_color