# This script runs the irreversible MASH dynamics and compare with the MASH implementation in SHARC
from analytical import *
from electronic import *
from engine import MASHEngineIrrev
from integrator import *
from trajectory import *
from util import *


### 1. Set up the initial condition ###

k = 0.1
vc = 0.01
x0 = 1.0
mass = 1836.15

DH_model = DoubleHarmonic3D(k=k, x0=x0, vc=vc, mass=mass)

q_init = np.array([-1.0, 0.0, 0.0])
v_init = np.array([0.0087, 0.0, 0.0])
coeff_init = np.array([1.0, 0.0], dtype=complex)
act_state_init = 0
dt = 20.67 # 0.5 femtoseconds
max_steps = 200
C_init = DH_model.eigvecs(q_init)

initial_snapshot = Snapshot(positions=q_init,
                            velocities=v_init,
                            coefficients=coeff_init,
                            active_state=act_state_init,
                            gauge=C_init,
                            mass=mass,
                            is_grid=True)

dynamics = MASHEngineIrrev(model=DH_model, dt=dt, initial_snapshot=initial_snapshot)

trajectory = dynamics.propagate(n_steps=max_steps)

positions = [trajectory[i].positions[0] * 0.529 for i in range(len(trajectory))]
for position in positions:
    print(float(position))




# The comparison test is concluded. SHARC and in-house code obtained very similar numerical results for the first few tens
# of steps before inevitably diverging due to differences in details of algorithms, numerical discrepancies, etc.