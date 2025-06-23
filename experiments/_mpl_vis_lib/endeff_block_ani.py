import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os


def animate(traj, box_params, filename):
    """
    animates the full trajecory of teh ee-obj system
    """
    # split up states
    x, xb = traj['x'], traj['xb']
    eepos = x[:,:2] 
    objpos = xb[:,:3]

    r_ee = 0.025

    # obtain parameters for plotting box shape
    l = ((0.5*box_params['l_box'])**2 + (0.5*box_params['w_box'])**2)**0.5
    beta = np.arctan(box_params['w_box']/box_params['l_box'])
    betas = [beta, np.pi-beta, np.pi+beta, 2*np.pi-beta]

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-1.0, 1.0)  
    ax.set_ylim(-1.0, 1.0)  
    ax.set_title("Contouring Experiment")
    ax.set_xlabel("1D Position")
    # ax.get_yaxis().set_visible(False)  

    # rotation in SO2
    def SO2_rotation_matrix(angle):
            """
            a simple method for SO(2) rotation, returns 2x2 matrix
            """
            return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    
    # Initialize markers for the objects
    ee = ax.add_patch(plt.Circle((eepos[0,0],eepos[0,1]), radius=r_ee, color='blue', label='ee'))
    vertices = []
    for beta in betas:
        vertices.append(tuple(
             (SO2_rotation_matrix(objpos[0,2]) @ np.array(
                  [
                    l*np.cos(beta), 
                    l*np.sin(beta)
                  ]
             ).reshape(2,1) 
             + np.array(objpos[0,0:2]).reshape(2,1)).flatten()
        ))
    polygon = Polygon(vertices, closed=True, edgecolor='k', facecolor='lightblue', alpha = 0.7) 
    ax.add_patch(polygon)
    ax.legend()

    def create_polygon(vertices):
            """Creates the vertices for an N-sided polygon."""
            x = []
            y = []
            for v in range(len(vertices)):
                x.append(vertices[v][0])
                y.append(vertices[v][1])
            
            return np.column_stack((x, y))
    
    # Initialization function
    def init():
        ee.set_center((eepos[0,0],eepos[0,1]))
        polygon.set_xy(create_polygon(vertices))
        return ee, polygon #, obj , obj_axis_x, obj_axis_y

    # Animation function
    def update(frame):
        ee.set_center((eepos[frame,0], eepos[frame,1]))  
        vertices = []
        for beta in betas:
            vertices.append(tuple(
                 (SO2_rotation_matrix(objpos[frame,2]) @ np.array(
                      [
                        l*np.cos(beta), 
                        l*np.sin(beta)
                      ]
                 ).reshape(2,1) 
                 + np.array(objpos[frame,:2]).reshape(2,1)).flatten()
            ))
        polygon.set_xy(create_polygon(vertices))
        return ee, polygon

    # Create animation
    ani = FuncAnimation(
        fig, update, frames=traj['x'].shape[0], init_func=init, blit=True, interval=10
    )

    # Create the folder (and parent directories if they don’t exist)
    os.makedirs(f'figs/shape_est', exist_ok=True)
    
    # save the animation
    ani.save(f'figs/shape_est/{filename}.mp4', writer='ffmpeg')

if __name__ == "__main__":
    # imports
    import dill as pkl

    with open(f'data/simdata.pkl', 'rb') as file:
        data = pkl.load(file)
    
    animate(data, {'l_box': 0.5, 'w_box': 0.25})