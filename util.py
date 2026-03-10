from trajectory import Trajectory

def remove_off_grid(traj: Trajectory) -> Trajectory:
    """
    removes those time steps from the trajectory 
    that do not coincide with multiple of dt
    """
    return Trajectory([snap for snap in traj if snap.is_grid])