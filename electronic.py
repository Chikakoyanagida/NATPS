import numpy as np
from typing import overload, Literal
from integrator import *


def lowdin_orthogonalization(S):
    """
    Compute T = S (S^T S)^(-1/2) as the Lowdin orthogonalisation process
    """

    A = S.conj().T @ S

    eigvals, eigvecs = np.linalg.eigh(A)

    eps = 1e-14
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))

    A_inv_sqrt = eigvecs @ inv_sqrt @ eigvecs.conj().T

    T = S @ A_inv_sqrt

    return T

def unitary_propagator(A, dt, hbar=1.0):
    """
    Compute U = exp(-i A dt / hbar)
    for Hermitian 2x2 matrix A.
    """

    # Diagonalize A
    eigvals, eigvecs = np.linalg.eigh(A)

    # Exponentiate eigenvalues
    phases = np.exp(-1j * eigvals * dt / hbar)

    # Reconstruct U
    U = eigvecs @ np.diag(phases) @ eigvecs.conj().T

    return U

def align_phases_real(C_prev, C_curr):
    """
    Enforce phase continuity between eigenvector matrices.
    Columns are eigenvectors.
    """
    C_aligned = C_curr.copy()

    for i in range(C_prev.shape[1]):
        overlap = np.vdot(C_prev[:, i], C_curr[:, i])
        if np.real(overlap) < 0:
            C_aligned[:, i] *= -1.0

    return C_aligned

def align_phases(C_prev, C_curr, eps=1e-14):
    """
    Enforce phase continuity between eigenvector matrices.
    Columns are eigenvectors.
    """
    C = C_curr.copy()
    for i in range(C_prev.shape[1]):
        ov = np.vdot(C_prev[:, i], C[:, i])  # <prev|curr>
        mag = np.abs(ov)
        if mag > eps:
            C[:, i] *= (ov.conjugate() / mag)  # multiply by e^{-i arg(ov)}
    return C

# Phase-tracking paper: set a reference phase to ensure global phase convention

def sz_from_coeff(c):
    '''
    Compute Sz from coefficients
    
    :param c: list (len2) of the coefficients in the two-state model
    '''
    return np.abs(c[1])**2 - np.abs(c[0])**2

def LD_propagate_CPA(model, q, dt, C_curr, coeff_init):
    '''
    Local diabatisation propagator for model Hamiltonians with Classical Path
    Approximation.
    
    :param model: class DiabaticTwoState1D
    :param q: Classical trajectory
    :param coeff_init: Initial state coefficient in adiabatic basis
    :param C_curr: Adiabatic state in diabatic basis
    '''
    coeff_store = [coeff_init]
    for i in range(len(q)-1):
        q_curr = q[i]
        q_next = q[i+1]
        C_next_raw = model.eigvecs(q=q_next)
        C_next = align_phases(C_curr, C_next_raw)
        En = np.diag([model.V(a=0, q=q_curr), model.V(a=1, q=q_curr)])
        Enp = np.diag([model.V(a=0, q=q_next), model.V(a=1, q=q_next)])
        Snp = C_curr.conj().T @ C_next
        T = lowdin_orthogonalization(Snp)
        T_inv = T.conj().T
        HLD = T @ Enp @ T_inv
        A = 0.5 * (En + HLD)
        U = unitary_propagator(A, dt)
        co_next = T_inv @ (U @ coeff_store[-1])
        coeff_store.append(co_next)
        C_curr = C_next
    return coeff_store

def LD_step(model, q_curr, q_next, dt, C_curr, coeff_curr):
    '''
    Step-wise local diabatisation 
    
    :param model: class DiabaticTwoState1D
    :param q_curr: current position
    :param q_next: next position
    :param dt: time step
    :param C_curr: current adiabatic basis
    :param coeff_curr: current coefficients
    '''
    if dt < 0:
        raise UserWarning('Negative timestep detected')
    C_next_raw = model.eigvecs(q=q_next)
    C_next = align_phases(C_curr, C_next_raw)
    En = np.diag([model.V(a=0, q=q_curr), model.V(a=1, q=q_curr)])
    Enp = np.diag([model.V(a=0, q=q_next), model.V(a=1, q=q_next)])
    Snp = C_curr.conj().T @ C_next
    T = lowdin_orthogonalization(Snp)
    T_inv = T.conj().T
    HLD = T @ Enp @ T_inv
    A = 0.5 * (En + HLD)
    U = unitary_propagator(A, dt)
    coeff_next = T_inv @ (U @ coeff_curr)
    return C_next, coeff_next

def local_diabatisation(model, snapshot, q_next, dt):
    q_curr = snapshot.positions
    C_curr = snapshot.gauge
    coeff_curr = snapshot.coefficients

    C_next_raw = model.eigvecs(q=q_next)
    C_next = align_phases(C_curr, C_next_raw)
    En = np.diag([model.V0(q=q_curr), model.V1(q=q_curr)])
    Enp = np.diag([model.V0(q=q_next), model.V1(q=q_next)])
    Snp = C_curr.conj().T @ C_next

    T = lowdin_orthogonalization(Snp)
    T_inv = T.conj().T
    HLD = T @ Enp @ T_inv
    A = 0.5 * (En + HLD)
    U = unitary_propagator(A, dt)
    coeff_next = T_inv @ (U @ coeff_curr)

    return C_next, coeff_next

def hop_search_bisect(model, dt, q_L, q_R, C_L, coeff_L, Sz_L, Sz_R, tol_tau=1e-6, tol_sz=1e-10, max_iter = 500):
    '''
    Use bisection search to find the exact moment when system undergoes a hop.
    
    :param model: model Hamiltonian in use
    :param dt: original timestep length
    :param q_L: starting position (of propagation)
    :param q_R: ending position
    :param C_L: starting adiabatic state (under diabatic representation)
    :param coeff_L: starting coefficients of the wavepacket (under adiabatic representation)
    :param Sz_L: starting spin z vector value
    :param Sz_R: ending spin z
    :param tol_tau: hop search convergence tolerance on hop time
    :param tol_sz: hop search convergence tolerance on Sz
    :param max_iter: Maximum iteration
    '''
    tau_L = 0.0
    tau_R = dt
    dq = q_R - q_L
    iter = 0
    # print(dq/dt) # In full dynamics replace with nuclear propagation

    while True:
        tau_M = 0.5 * (tau_L + tau_R)
        q_M = q_L + tau_M * (dq/dt) # In full dynamics replace with nuclear propagation
        C_M, coeff_M = LD_step(model, q_L, q_M, tau_M, C_L, coeff_L)
        Sz_M = sz_from_coeff(coeff_M)
        if Sz_L * Sz_M < 0:
            tau_R = tau_M
            Sz_R = Sz_M
        elif Sz_M * Sz_R < 0:
            tau_L = tau_M
            Sz_L = Sz_M
        
        if (np.abs(tau_R-tau_L) < tol_tau) or (np.abs(Sz_M) < tol_sz):
            break

        iter += 1
        if iter > max_iter:
            raise RuntimeError('Bisection search did not converge')
    
    print('Hop search finished after', iter, 'iterations')
    
    tau_star = 0.5 * (tau_L + tau_R)
    q_star = q_L + tau_star * (dq/dt)
    C_star, coeff_star = LD_step(model, q_L, q_star, tau_star, C_L, coeff_L)
        
    return tau_star, q_star, C_star, coeff_star

# Why is there no quasi-analytical appraoch to pinpoint the hopping? Ask Jeremy. Don't ask Jeremy

# Todo: enforce a global gauge convention to remove gauge dependency.

def hop_search_direct(model, dt, q_L, v_L, C_L, Sz_L, Sz_R, coeff_L, tol_tau = 1e-10, tol_sz = 1e-12, max_iter = 500):
    tau_L = 0.0
    tau_R = dt
    iter = 0
    active_state = 1 if Sz_L > 0 else 0
    
    
    while True:
        F_L = model.F(a=active_state, q=q_L)
        tau_M = 0.5 * (tau_L + tau_R)
        v_M_half = verlet_v(tau_M, v_L, F_L)
        q_M = q_L + tau_M * v_M_half
        F_M = model.F(a=active_state, q=q_M)
        v_M = verlet_v(tau_M, v_M_half, F_M)
        C_M, coeff_M = LD_step(model, q_L, q_M, tau_M, C_L, coeff_L)
        Sz_M = sz_from_coeff(coeff_M)
        if Sz_L * Sz_M < 0:
            tau_R = tau_M
            Sz_R = Sz_M
        elif Sz_M * Sz_R < 0:
            tau_L = tau_M
            Sz_L = Sz_M
        
        if (np.abs(tau_R-tau_L) < tol_tau) or (np.abs(Sz_M) < tol_sz):
            break
        iter += 1
        if iter > max_iter:
            raise RuntimeError('Bisection search did not converge')
        
    print('Hop search finished after', iter, 'iterations')
    tau_star = tau_M
    q_star = q_M
    v_star = v_M
    C_star, coeff_star = C_M, coeff_M
    return tau_star, q_star, v_star, C_star, coeff_star

def hop_search(dt, snapshot, model, tol_tau=1e-8, tol_Sz=1e-10, max_iter=500):
    
    coeff_curr = snapshot.coefficients
    mass = snapshot.mass
    active_state = snapshot.active_state
    q_curr = snapshot.positions
    v_curr = snapshot.velocities

    tau_L = 0.0
    tau_R = dt
    Sz_curr = sz_from_coeff(coeff_curr)
    iter = 0

    while True:
        tau_M = 0.5 * (tau_L + tau_R)
        v_half = verlet_v(dt=tau_M, model=model, q_curr=q_curr, v_curr=v_curr, active_state=active_state, mass=mass)
        q_next = verlet_X(dt=tau_M, q_curr=q_curr, v_half=v_half)
        C_next, coeff_next = local_diabatisation(snapshot=snapshot, model=model, q_next=q_next, dt=tau_M)
        Sz_next = sz_from_coeff(coeff_next)

        if abs(Sz_next) < tol_Sz or abs(tau_L-tau_R) < tol_tau:
            break

        if Sz_curr * Sz_next < 0:
            tau_R = tau_M
        else:
            tau_L = tau_M
        
        iter += 1
        if iter > max_iter:
            raise RuntimeError('Maximum iteration reached')
    print('Hop search finished after', iter, 'iterations')
    
    return tau_M
