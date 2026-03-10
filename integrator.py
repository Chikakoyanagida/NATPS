import numpy as np
from typing import Tuple

from analytical import *
from trajectory import Snapshot

def verlet_X(
    dt: float, q_curr: np.ndarray, v_half: np.ndarray
) -> np.ndarray:
    """Velocity Verlet position update: q_next = q_curr + dt * v_half."""
    return q_curr + dt * v_half


def verlet_v(
    dt: float,
    model: DiabaticTwoState1D,
    mass: float,
    v_curr: np.ndarray,
    q_curr: np.ndarray,
    active_state: int,
) -> np.ndarray:
    """Velocity Verlet velocity half-step: v_half = v_curr + 0.5 * dt * F/mass."""
    F_curr = model.F(a=active_state, q=q_curr)
    return v_curr + 0.5 * dt * F_curr / mass


def velocity_rescaling(
    model: DiabaticTwoState1D, snapshot: Snapshot
) -> Tuple[np.ndarray, bool]:
    """Surface hopping velocity rescaling with momentum adjustment."""
    mass = snapshot.mass
    active_state = snapshot.active_state
    v_old = snapshot.velocities
    q_hop = snapshot.positions

    d10 = model.d_an(q=q_hop, a=1)
    adiabatic_energies = [model.V0(q=q_hop), model.V1(q=q_hop)]
    is_hop = False

    V_init = adiabatic_energies[active_state]
    V_fin = adiabatic_energies[1 - active_state]

    mw_p_old = np.sqrt(mass) * v_old
    mw_nac = d10 / np.sqrt(mass)

    mw_p_old_d = np.dot(mw_nac, mw_p_old) / np.abs(mw_nac)
    E_d = 0.5 * mw_p_old_d**2 + V_init

    if E_d >= V_fin:
        mw_p_new_d = np.sign(mw_p_old_d) * np.sqrt(
            max(0.0, mw_p_old_d**2 + 2 * (V_init - V_fin))
        )
        delta_d = mw_p_old_d - mw_p_new_d
        mw_p_new = mw_p_old - (mw_nac / np.abs(mw_nac)) * delta_d
        is_hop = True
    else:
        mw_p_new = mw_p_old - 2 * (mw_nac / np.abs(mw_nac)) * mw_p_old_d
        is_hop = False

    v_new = mw_p_new / np.sqrt(mass)
    return v_new, is_hop
