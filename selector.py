import numpy as np
import copy
from electronic import *

class UniformShootingSelector:
    
    def __init__(self, temp, mass, stateA_func, stateB_func, alpha=0.5):
        kB_au = 3.1668114e-6
        self.kbT = kB_au * temp
        self.mass = mass
        self.alpha = alpha
        self.inA = stateA_func
        self.inB = stateB_func

    def _is_eligible(self, snap):

        if not snap.is_grid:
            return False
            
        in_A = self.inA(snap.positions, snap.active_state)
        in_B = self.inB(snap.positions, snap.active_state)
        
        return not (in_A or in_B)
    
    def select_and_perturb(self, traj):
        
        eligible_snaps = [snap for snap in traj if self._is_eligible(snap)]
        l_old = len(eligible_snaps)
        if l_old <= 0:
            raise ValueError("Trajectory too short to shoot after padding!")

        shooting_index = np.random.randint(0, l_old)
        shooting_snapshot = copy.deepcopy(eligible_snaps[shooting_index])

        v_old = shooting_snapshot.velocities
        v_mb = np.random.normal(
            loc=0.0,
            scale=np.sqrt(self.kbT / self.mass),
        )
        v_new = self.alpha * v_old + np.sqrt(1.0 - self.alpha**2) * v_mb

        shooting_snapshot.velocities = v_new

        return shooting_snapshot, l_old
    
    def check_acceptance(self, l_old, trial_traj, connected):
        if not connected:
            return False
        
        l_new = sum(1 for snap in trial_traj if self._is_eligible(snap))
        if l_new == 0:
            return False
        
        P_acc = min(1.0, l_old / l_new)
        rand_val = np.random.uniform(0.0, 1.0)
        return bool(rand_val < P_acc)
    
