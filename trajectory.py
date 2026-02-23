import numpy as np
from electronic import *

class BaseSnapshot:
    def __init__(self, positions, velocities, coefficients, gauge):
        self.positions = positions
        self.velocities = velocities
        self.coefficients = coefficients
        self.gauge = gauge
    
    @property
    def active_state(self,):
        Sz = sz_from_coeff(self.coefficients)
        if Sz > 0:
            return 1
        else:
            return 0