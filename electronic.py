import numpy as np
from typing import List, Tuple

from integrator import verlet_v, verlet_X  
from analytical import DiabaticTwoState1D
from trajectory import Snapshot


def lowdin_orthogonalization(S: np.ndarray) -> np.ndarray:
    """
    Perform Löwdin symmetric orthogonalization on an overlap matrix.

    Given an overlap matrix S, compute the transformation
    T = S (S† S)^(-1/2), such that T produces an orthonormal basis
    closest to the original in least-squares sense.

    Args:
        S: Overlap matrix of shape (n, n), generally complex.

    Returns:
        Orthogonalization matrix T of shape (n, n).
    """
    gram = S.conj().T @ S

    eigvals, eigvecs = np.linalg.eigh(gram)

    eps = 1e-14
    inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))

    gram_inv_sqrt = eigvecs @ inv_sqrt @ eigvecs.conj().T
    T = S @ gram_inv_sqrt
    return T


def unitary_propagator(
    A: np.ndarray,
    dt: float,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Compute the unitary time-evolution operator U = exp(-i A dt / hbar).

    Args:
        A: Hermitian Hamiltonian matrix, typically 2x2.
        dt: Time step.
        hbar: Reduced Planck constant (default 1.0).

    Returns:
        Unitary propagator matrix U with the same shape as A.
    """
    eigvals, eigvecs = np.linalg.eigh(A)
    phases = np.exp(-1j * eigvals * dt / hbar)
    U = eigvecs @ np.diag(phases) @ eigvecs.conj().T
    return U


def align_phases_real(c_prev: np.ndarray, c_curr: np.ndarray) -> np.ndarray:
    """
    Align phases of eigenvectors so their overlaps are real and positive.

    For each column i, if Re(<C_prev_i|C_curr_i>) < 0, flip the sign of C_curr_i.[web:59]

    Args:
        C_prev: Previous eigenvector matrix, shape (n, m).
        C_curr: Current eigenvector matrix, shape (n, m).

    Returns:
        Phase-aligned eigenvector matrix with same shape as C_curr.
    """
    c_aligned = c_curr.copy()

    for i in range(c_prev.shape[1]):
        overlap = np.vdot(C_prev[:, i], c_curr[:, i])
        if np.real(overlap) < 0.0:
            c_aligned[:, i] *= -1.0

    return c_aligned


def align_phases(
    c_prev: np.ndarray,
    c_curr: np.ndarray,
    eps: float = 1e-14,
) -> np.ndarray:
    """
    Align phases of eigenvectors for continuous complex phases.

    For each column i with non-negligible overlap ov = <prev|curr>,
    multiply C_curr_i by ov* / |ov| so that the new overlap is real and positive.

    Args:
        C_prev: Previous eigenvector matrix, shape (n, m).
        C_curr: Current eigenvector matrix, shape (n, m).
        eps: Threshold below which overlaps are considered zero.

    Returns:
        Phase-aligned eigenvector matrix with same shape as C_curr.
    """
    c_aligned = c_curr.copy()
    for i in range(c_prev.shape[1]):
        overlap = np.vdot(c_prev[:, i], c_aligned[:, i])
        magnitude = np.abs(overlap)
        if magnitude > eps:
            c_aligned[:, i] *= overlap.conjugate() / magnitude
    return c_aligned


def sz_from_coeff(c: np.ndarray) -> float:
    """
    Compute Sz from two-state coefficients on adiabatic surfaces.

    For coefficients c = (c0, c1), return
        Sz = |c1|^2 − |c0|^2,
    i.e., the z-component of the Bloch vector in a two-level system.

    Args:
        c: Complex coefficient vector of length 2.

    Returns:
        Scalar Sz value.
    """
    return float(np.abs(c[1]) ** 2 - np.abs(c[0]) ** 2)


def LD_propagate_CPA(
    model: DiabaticTwoState1D,
    q: np.ndarray,
    dt: float,
    C_curr: np.ndarray,
    coeff_init: np.ndarray,
) -> List[np.ndarray]:
    """
    Local diabatization propagation under the Classical Path Approximation.

    Propagates electronic coefficients along a fixed classical trajectory q(t),
    using Löwdin orthogonalization and a locally diabatic Hamiltonian.[web:59][web:64]

    Args:
        model: Diabatic two-state 1D model providing eigvecs and V(a, q).
        q: Array of classical positions along the trajectory, shape (n_steps,).
        dt: Time step between consecutive positions.
        C_curr: Initial adiabatic eigenvectors in diabatic basis, shape (2, 2).
        coeff_init: Initial adiabatic coefficients, shape (2,).

    Returns:
        List of coefficient vectors at each step (length n_steps).
    """
    coeff_store: List[np.ndarray] = [coeff_init]

    for i in range(len(q) - 1):
        q_curr = q[i]
        q_next = q[i + 1]

        C_next_raw = model.eigvecs(q=q_next)
        C_next = align_phases(C_curr, C_next_raw)

        En = np.diag([model.V(a=0, q=q_curr), model.V(a=1, q=q_curr)])
        Enp = np.diag([model.V(a=0, q=q_next), model.V(a=1, q=q_next)])

        Snp = C_curr.conj().T @ C_next
        T = lowdin_orthogonalization(Snp)
        T_inv = T.conj().T

        H_local_diab = T @ Enp @ T_inv
        A_eff = 0.5 * (En + H_local_diab)

        U = unitary_propagator(A_eff, dt)
        coeff_next = T_inv @ (U @ coeff_store[-1])

        coeff_store.append(coeff_next)
        C_curr = C_next

    return coeff_store


def LD_step(
    model: DiabaticTwoState1D,
    q_curr: float,
    q_next: float,
    dt: float,
    C_curr: np.ndarray,
    coeff_curr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single local diabatization step between two nuclear positions.

    Args:
        model: Diabatic two-state 1D model.
        q_curr: Current nuclear position.
        q_next: Next nuclear position.
        dt: Time step between q_curr and q_next (must be non-negative).
        C_curr: Current adiabatic eigenvectors in diabatic basis, shape (2, 2).
        coeff_curr: Current adiabatic coefficients, shape (2,).

    Returns:
        Tuple (C_next, coeff_next):
            C_next: Next adiabatic eigenvectors (phase-aligned), shape (2, 2).
            coeff_next: Next adiabatic coefficients, shape (2,).
    """
    if dt < 0.0:
        raise UserWarning("Negative timestep detected")

    C_next_raw = model.eigvecs(q=q_next)
    C_next = align_phases(C_curr, C_next_raw)

    En = np.diag([model.V(a=0, q=q_curr), model.V(a=1, q=q_curr)])
    Enp = np.diag([model.V(a=0, q=q_next), model.V(a=1, q=q_next)])

    Snp = C_curr.conj().T @ C_next
    T = lowdin_orthogonalization(Snp)
    T_inv = T.conj().T

    H_local_diab = T @ Enp @ T_inv
    A_eff = 0.5 * (En + H_local_diab)

    U = unitary_propagator(A_eff, dt)
    coeff_next = T_inv @ (U @ coeff_curr)

    return C_next, coeff_next


def local_diabatisation(
    model: DiabaticTwoState1D,
    snapshot: Snapshot,
    q_next: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local diabatization step based on a Snapshot object.

    Args:
        model: Diabatic two-state 1D model.
        snapshot: Current nuclear and electronic state.
        q_next: Next nuclear position.
        dt: Time step between snapshot.positions and q_next.

    Returns:
        Tuple (C_next, coeff_next):
            C_next: Next adiabatic eigenvectors (phase-aligned), shape (2, 2).
            coeff_next: Next adiabatic coefficients, shape (2,).
    """
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
    H_local_diab = T @ Enp @ T_inv
    A_eff = 0.5 * (En + H_local_diab)

    U = unitary_propagator(A_eff, dt)
    coeff_next = T_inv @ (U @ coeff_curr)

    return C_next, coeff_next


def hop_search_bisect(
    model: DiabaticTwoState1D,
    dt: float,
    q_L: float,
    q_R: float,
    C_L: np.ndarray,
    coeff_L: np.ndarray,
    Sz_L: float,
    Sz_R: float,
    tol_tau: float = 1e-6,
    tol_sz: float = 1e-10,
    max_iter: int = 500,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Locate a hopping time by bisection using Sz sign change along a fixed path.

    Args:
        model: Model Hamiltonian.
        dt: Original time step.
        q_L: Left nuclear position.
        q_R: Right nuclear position.
        C_L: Adiabatic basis at q_L, shape (2, 2).
        coeff_L: Coefficients at q_L, shape (2,).
        Sz_L: Sz value at q_L.
        Sz_R: Sz value at q_R.
        tol_tau: Tolerance on hop time.
        tol_sz: Tolerance on Sz at the hop.
        max_iter: Maximum number of bisection iterations.

    Returns:
        Tuple (tau_star, q_star, C_star, coeff_star):
            tau_star: Estimated hop time within [0, dt].
            q_star: Nuclear position at the hop.
            C_star: Adiabatic basis at q_star, shape (2, 2).
            coeff_star: Coefficients at q_star, shape (2,).
    """
    tau_L = 0.0
    tau_R = dt
    dq = q_R - q_L
    iteration = 0

    while True:
        tau_M = 0.5 * (tau_L + tau_R)
        q_M = q_L + tau_M * (dq / dt)
        C_M, coeff_M = LD_step(model, q_L, q_M, tau_M, C_L, coeff_L)
        Sz_M = sz_from_coeff(coeff_M)

        if Sz_L * Sz_M < 0.0:
            tau_R = tau_M
            Sz_R = Sz_M
        elif Sz_M * Sz_R < 0.0:
            tau_L = tau_M
            Sz_L = Sz_M

        if (np.abs(tau_R - tau_L) < tol_tau) or (np.abs(Sz_M) < tol_sz):
            break

        iteration += 1
        if iteration > max_iter:
            raise RuntimeError("Bisection search did not converge")

    print("Hop search finished after", iteration, "iterations")

    tau_star = 0.5 * (tau_L + tau_R)
    q_star = q_L + tau_star * (dq / dt)
    C_star, coeff_star = LD_step(model, q_L, q_star, tau_star, C_L, coeff_L)

    return tau_star, q_star, C_star, coeff_star

# Todo: enforce a global gauge convention to remove gauge dependency.

def hop_search_direct(
    model: DiabaticTwoState1D,
    dt: float,
    q_L: float,
    v_L: float,
    C_L: np.ndarray,
    Sz_L: float,
    Sz_R: float,
    coeff_L: np.ndarray,
    tol_tau: float = 1e-10,
    tol_sz: float = 1e-12,
    max_iter: int = 500,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Direct hop search with Verlet nuclear propagation and Sz sign change.

    Args:
        model: Model Hamiltonian.
        dt: Original time step.
        q_L: Initial nuclear position.
        v_L: Initial nuclear velocity.
        C_L: Initial adiabatic basis at q_L, shape (2, 2).
        Sz_L: Sz at left endpoint.
        Sz_R: Sz at right endpoint.
        coeff_L: Initial coefficients at q_L, shape (2,).
        tol_tau: Tolerance on hop time.
        tol_sz: Tolerance on Sz at the hop.
        max_iter: Maximum number of iterations.

    Returns:
        Tuple (tau_star, q_star, v_star, C_star, coeff_star):
            tau_star: Estimated hop time.
            q_star: Nuclear position at hop.
            v_star: Nuclear velocity at hop.
            C_star: Adiabatic basis at hop.
            coeff_star: Coefficients at hop.
    """
    tau_L = 0.0
    tau_R = dt
    iteration = 0
    active_state = 1 if Sz_L > 0.0 else 0

    while True:
        F_L = model.F(a=active_state, q=q_L)
        tau_M = 0.5 * (tau_L + tau_R)

        v_M_half = verlet_v(tau_M, v_L, F_L)
        q_M = q_L + tau_M * v_M_half
        F_M = model.F(a=active_state, q=q_M)
        v_M = verlet_v(tau_M, v_M_half, F_M)

        C_M, coeff_M = LD_step(model, q_L, q_M, tau_M, C_L, coeff_L)
        Sz_M = sz_from_coeff(coeff_M)

        if Sz_L * Sz_M < 0.0:
            tau_R = tau_M
            Sz_R = Sz_M
        elif Sz_M * Sz_R < 0.0:
            tau_L = tau_M
            Sz_L = Sz_M

        if (np.abs(tau_R - tau_L) < tol_tau) or (np.abs(Sz_M) < tol_sz):
            break

        iteration += 1
        if iteration > max_iter:
            raise RuntimeError("Bisection search did not converge")

    print("Hop search finished after", iteration, "iterations")
    tau_star = tau_M
    q_star = q_M
    v_star = v_M
    C_star, coeff_star = C_M, coeff_M
    return tau_star, q_star, v_star, C_star, coeff_star


def hop_search(
    dt: float,
    snapshot: Snapshot,
    model: DiabaticTwoState1D,
    tol_tau: float = 1e-8,
    tol_Sz: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """
    Hop search using internal Verlet propagation and local diabatization.

    Performs a bisection-like search in time to locate a hop where Sz changes sign.

    Args:
        dt: Original time step.
        snapshot: Current nuclear and electronic state.
        model: Diabatic two-state 1D model.
        tol_tau: Tolerance on hop time.
        tol_Sz: Tolerance on Sz at the hop.
        max_iter: Maximum number of iterations.

    Returns:
        Estimated hop time tau_M within [0, dt].
    """
    coeff_curr = snapshot.coefficients
    mass = snapshot.mass
    active_state = snapshot.active_state
    q_curr = snapshot.positions
    v_curr = snapshot.velocities

    tau_L = 0.0
    tau_R = dt
    Sz_curr = sz_from_coeff(coeff_curr)
    iteration = 0

    while True:
        tau_M = 0.5 * (tau_L + tau_R)

        v_half = verlet_v(
            dt=tau_M,
            model=model,
            mass=mass,
            v_curr=v_curr,
            q_curr=q_curr,
            active_state=active_state,
        )
        q_next = verlet_X(dt=tau_M, q_curr=q_curr, v_half=v_half)

        _, coeff_next = local_diabatisation(
            snapshot=snapshot,
            model=model,
            q_next=q_next,
            dt=tau_M,
        )
        Sz_next = sz_from_coeff(coeff_next)

        if abs(Sz_next) < tol_Sz or abs(tau_L - tau_R) < tol_tau:
            break

        if Sz_curr * Sz_next < 0.0:
            tau_R = tau_M
        else:
            tau_L = tau_M

        iteration += 1
        if iteration > max_iter:
            raise RuntimeError("Maximum iteration reached")
    #print(
    #    "Hop search finished after",
    #    iteration,
    #    "iterations. Hop is located at",
    #    f"x={q_next}.",
    #)

    return tau_M
