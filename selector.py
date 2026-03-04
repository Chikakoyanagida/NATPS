import numpy as np
import copy
from electronic import *

class UniformShootingSelector:
    
    def __init__(self, temp, mass, alpha=0.5):
        kB_au = 3.1668114e-6
        self.kbT = kB_au * temp
        self.mass = mass
        self.alpha = alpha
    
    def select_and_perturb(self, traj, pad_L=1, pad_R=1):
        
        grid_traj = [snap for snap in traj if snap.is_grid]
        L_old = len(grid_traj)
        l_old = L_old - pad_L - pad_R
        if l_old <= 0:
            raise ValueError('Trajectory too short to shoot after padding!')
        

        shooting_index = np.random.randint(pad_L, L_old-pad_R)
        shooting_snapshot = copy.deepcopy(grid_traj[shooting_index])

        v_old = shooting_snapshot.velocities
        v_mb = np.random.normal(loc=0.0, scale=np.sqrt(self.kbT / self.mass))
        v_new = self.alpha * v_old + np.sqrt(1-self.alpha**2) * v_mb

        shooting_snapshot.velocities = v_new

        return shooting_snapshot, l_old
    
    def check_acceptance(self, l_old, l_new, connected):
        if not connected:
            return False
        
        P_acc = min(1.0, l_old/l_new)
        ran = np.random.uniform(0.0, 1.0)
        chance = ran < P_acc

        return chance
    
