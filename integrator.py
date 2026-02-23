import numpy as np

def verlet_X(dt, q_curr, v_half):
    q_next = q_curr + dt * v_half
    return q_next

def verlet_v(dt, v_curr, F_curr):
    v_half = v_curr + 0.5 * dt * F_curr
    return v_half