import numpy as np

def verlet_X(dt, q_curr, v_half):
    q_next = q_curr + dt * v_half
    return q_next

def verlet_v(dt, v_curr, F_curr, model):
    mass = model.mass
    v_half = v_curr + 0.5 * dt * F_curr/mass
    return v_half

def surface_hop(Sz_prev, S_next, q_curr, v_curr, model):
    mass = model.mass
    hop_exists = False
    if Sz_prev > 0:
        V_init = model.V1(q=q_curr)
        V_fin = model.V0(q=q_curr)
    elif Sz_prev < 0:
        V_init = model.V0(q=q_curr)
        V_fin = model.V1(q=q_curr)
        
    mw_mo = np.sqrt(mass) * v_curr
    d10 = model.d_an(q=q_curr, a=1)
    p_d_init = np.sign(d10) * mw_mo
    Ed = 0.5 * p_d_init**2 + V_init
    if Ed >= V_fin:
        hop_exists = True
        p_d_fin = np.sign(p_d_init) * np.sqrt(p_d_init**2 + 2.0 * (V_init - V_fin))
        mw_mo_fin = mw_mo - np.sign(d10) * (p_d_init - p_d_fin)
        Sz_reflected = S_next
    if Ed < V_fin:
        mw_mo_fin = mw_mo - 2.0 * np.sign(d10) * p_d_init
        Sz_reflected = -S_next
    
    v_rescaled = mw_mo_fin / np.sqrt(mass)

    return hop_exists, v_rescaled, Sz_reflected