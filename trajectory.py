import numpy as np
from electronic import *
import copy

class Snapshot:
    def __init__(self, positions, velocities, coefficients, active_state, gauge, mass, is_grid, hop=None):
        self.positions = positions
        self.velocities = velocities
        self.coefficients = coefficients
        self.gauge = gauge
        self.active_state = active_state
        self.mass = mass
        self.is_grid = is_grid
        self.hop = hop
    
    def reversed(self, ):
        target = copy.deepcopy(self)
        target.velocities *= -1.0
        target.coefficients = np.conjugate(target.coefficients)
        return target
    

class Trajectory:
    def __init__(self, states=None):
        """
        states : iterable of trajectory states (positions, velocities, etc.)
        """
        if states is None:
            self._states = []
        else:
            self._states = list(states)

    def __len__(self):
        """
        Enables len(traj)
        """
        return len(self._states)

    def __getitem__(self, index):
        """
        Enables traj[i]
        """
        return self._states[index]

    def append(self, state):
        """
        Add a new state to trajectory
        """
        self._states.append(state)

    def __repr__(self):
        return f"Trajectory(length={len(self)})"