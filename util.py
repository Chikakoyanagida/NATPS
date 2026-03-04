from trajectory import *

def remove_off_grid(traj):
    return Trajectory([snap for snap in traj if snap.is_grid])