import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

HBAR = 1.0

class DiabaticTwoState1D(ABC):
    """
    Base class for a 1D two-state system defined in a diabatic basis.

    Required overrides:
        H(q)     -> 2x2 diabatic Hamiltonian matrix
        dH_dq(q) -> 2x2 derivative of Hamiltonian wrt q

    Provides:
        - Adiabatic energies and NACs
        - Forces on adiabatic surfaces
        - MASH integrator interface
    """

    def __init__(self, mass: float = 1.0) -> None:
        self.mass = float(mass)

    @abstractmethod
    def H(self, q: float) -> np.ndarray:
        """2x2 diabatic Hamiltonian at position q."""
        pass
    
    @abstractmethod
    def dH_dq(self, q: float) -> np.ndarray:
        """2x2 derivative dH/dq at position q."""
        pass

    # ---- internal helper: adiabatic info at q ----
    def adiabatic_info(self, q: float) -> Tuple[float, float, float, float, float, np.ndarray]:
        """Compute adiabatic energies, gradients, NAC d10, and eigenvectors at q."""
        H = self.H(q)
        dH = self.dH_dq(q)

        eigvals, eigvecs = np.linalg.eigh(H)
        idx = np.argsort(eigvals)
        V0, V1 = eigvals[idx]
        C = eigvecs[:, idx]

        dV0_dq = np.real(np.vdot(C[:, 0], dH @ C[:, 0]))
        dV1_dq = np.real(np.vdot(C[:, 1], dH @ C[:, 1]))

        num_10 = np.vdot(C[:, 1], dH @ C[:, 0])   # <1| dH/dq |0>
        denom_10 = V0 - V1

        # regularise NAC near degeneracy
        eps = 1e-8
        if abs(denom_10) < eps:
            d10 = 0.0   # or smoothly clamp, but zero is fine for a start
        else:
            d10 = np.real(num_10 / denom_10)

        return V0, V1, dV0_dq, dV1_dq, d10, C

    def V0(self, q: float) -> float:
        """Adiabatic ground state energy V0(q)."""
        V0, _, _, _, _, _ = self.adiabatic_info(q)
        return V0

    def V1(self, q: float) -> float:
        """Adiabatic excited state energy V1(q)."""
        _, V1, _, _, _, _ = self.adiabatic_info(q)
        return V1

    def V(self, a: int, q: float) -> float:
        """Adiabatic energy on state a (0 or 1)."""
        if a == 0:
            return self.V0(q)
        elif a == 1:
            return self.V1(q)
        else:
            raise ValueError("State index a must be 0 or 1.")

    def F(self, a: int, q: float) -> float:
        """Adiabatic force F_a(q) = -dV_a/dq on state a (0 or 1)."""
        _, _, dV0_dq, dV1_dq, _, _ = self.adiabatic_info(q)
        if a == 0:
            return -dV0_dq
        elif a == 1:
            return -dV1_dq
        else:
            raise ValueError("State index a must be 0 or 1.")

    def d_an(self, q: float, a: int) -> float:
        """
        Nonadiabatic coupling d_{a n}(q) with n = 1 - a.

        Convention from adiabatic_info:
            d10 = <1|∂/∂q|0>

        For a 2-state, real-valued gauge, we have:
            d01 = -d10

        We return d_{a n} consistently:

            if a = 0, n = 1: d_{0 1} = -d10
            if a = 1, n = 0: d_{1 0} = d10
        """
        _, _, _, _, d10, _ = self.adiabatic_info(q)
        if a == 0:
            # a=0, n=1: d_01 = -d_10
            return -d10
        elif a == 1:
            # a=1, n=0: d_10
            return d10
        else:
            raise ValueError("State index a must be 0 or 1.")
    
    def eigvecs(self, q: float):
        """Adiabatic eigenvectors at q (sorted by energy)."""
        _, _, _, _, _, eigvecs = self.adiabatic_info(q)
        return eigvecs

class DiabaticTwoState3D(ABC):
    """
    Base class for a 3D two-state system defined in a diabatic basis.

    Required overrides:
        H(q)     -> 2x2 diabatic Hamiltonian matrix
        dH_dq(q) -> 2x2 derivative of Hamiltonian wrt q

    Provides:
        - Adiabatic energies and NACs
        - Forces on adiabatic surfaces
        - MASH integrator interface
    """
    def __init__(self, mass: float = 1.0) -> None:
        self.mass = float(mass)

    @abstractmethod
    def H(self, q: np.ndarray) -> np.ndarray:
        """2x2 diabatic Hamiltonian at position q."""
        pass
    
    @abstractmethod
    def dH_dq(self, q: np.ndarray) -> np.ndarray:
        """2x2 gradient \nabla H at position q."""
        pass
    
    # ---- internal helper: adiabatic info at q ----
    def adiabatic_info(self, q: np.ndarray):
        """Compute adiabatic energies, gradients, NAC d10, and eigenvectors at q."""
        H = self.H(q)
        dH = self.dH_dq(q)
        dHx = dH[0]
        dHy = dH[1]
        dHz = dH[2]

        eigvals, eigvecs = np.linalg.eigh(H)
        idx = np.argsort(eigvals)
        V0, V1 = eigvals[idx]
        C = eigvecs[:, idx]

        dV0_dqx = np.real(np.vdot(C[:, 0], dHx @ C[:, 0]))
        dV1_dqx = np.real(np.vdot(C[:, 1], dHx @ C[:, 1]))

        dV0_dqy = np.real(np.vdot(C[:, 0], dHy @ C[:, 0]))
        dV1_dqy = np.real(np.vdot(C[:, 1], dHy @ C[:, 1]))

        dV0_dqz = np.real(np.vdot(C[:, 0], dHz @ C[:, 0]))
        dV1_dqz = np.real(np.vdot(C[:, 1], dHz @ C[:, 1]))

        dV0_dq = np.array([dV0_dqx, dV0_dqy, dV0_dqz])

        dV1_dq = np.array([dV1_dqx, dV1_dqy, dV1_dqz])

        num_10x = np.vdot(C[:, 1], dHx @ C[:, 0])   # <1| dH/dq |0>
        num_10y = np.vdot(C[:, 1], dHy @ C[:, 0])
        num_10z = np.vdot(C[:, 1], dHz @ C[:, 0])

        num_10 = np.array([num_10x, num_10y, num_10z])
        denom_10 = V0 - V1

        # regularise NAC near degeneracy
        eps = 1e-8
        if abs(denom_10) < eps:
            d10 = np.zeros(3)   # or smoothly clamp, but zero is fine for a start
        else:
            d10 = np.real(num_10 / denom_10)

        return V0, V1, dV0_dq, dV1_dq, d10, C
    
    def V0(self, q: np.ndarray) -> float:
        """Adiabatic ground state energy V0(q)."""
        V0, _, _, _, _, _ = self.adiabatic_info(q)
        return V0

    def V1(self, q: np.ndarray) -> float:
        """Adiabatic excited state energy V1(q)."""
        _, V1, _, _, _, _ = self.adiabatic_info(q)
        return V1

    def V(self, a: int, q: np.ndarray) -> float:
        """Adiabatic energy on state a (0 or 1)."""
        if a == 0:
            return self.V0(q)
        elif a == 1:
            return self.V1(q)
        else:
            raise ValueError("State index a must be 0 or 1.")

    def F(self, a: int, q: np.ndarray) -> np.ndarray:
        """Adiabatic force F_a(q) = -dV_a/dq on state a (0 or 1)."""
        _, _, dV0_dq, dV1_dq, _, _ = self.adiabatic_info(q)
        if a == 0:
            return -dV0_dq
        elif a == 1:
            return -dV1_dq
        else:
            raise ValueError("State index a must be 0 or 1.")

    def d_an(self, q: np.ndarray, a: int) -> np.ndarray:
        """
        Nonadiabatic coupling d_{a n}(q) with n = 1 - a.

        Convention from adiabatic_info:
            d10 = <1|∂/∂q|0>

        For a 2-state, real-valued gauge, we have:
            d01 = -d10

        We return d_{a n} consistently:

            if a = 0, n = 1: d_{0 1} = -d10
            if a = 1, n = 0: d_{1 0} = d10
        """
        _, _, _, _, d10, _ = self.adiabatic_info(q)
        if a == 0:
            # a=0, n=1: d_01 = -d_10
            return -d10
        elif a == 1:
            # a=1, n=0: d_10
            return d10
        else:
            raise ValueError("State index a must be 0 or 1.")
    
    def eigvecs(self, q: np.ndarray):
        """Adiabatic eigenvectors at q (sorted by energy)."""
        _, _, _, _, _, eigvecs = self.adiabatic_info(q)
        return eigvecs

    

class LandauZener(DiabaticTwoState1D):
    """Landau-Zener model: two linear diabatic surfaces."""

    def __init__(self, s: float, vc: float, mass: float = 1.0) -> None:
        super().__init__(mass)
        self.s = s
        self.vc = vc

    def H(self, q: float) -> np.ndarray:
        V11 = 0.0
        V12 = self.vc
        V22 = self.s * q
        return np.array([[V11, V12], [V12, V22]], dtype=float)

    def dH_dq(self, q: float) -> np.ndarray:
        dV11 = 0.0
        dV12 = 0.0
        dV22 = self.s
        return np.array([[dV11, dV12], [dV12, dV22]], dtype=float)


class DoubleHarmonic(DiabaticTwoState1D):
    """Two horizontally displaced harmonic oscillators."""

    def __init__(self, k: float, x0: float, vc: float, mass: float = 1.0) -> None:
        super().__init__(mass)
        self.k = k
        self.x0 = x0
        self.vc = vc

    def H(self, q: float) -> np.ndarray:
        V11 = 0.5 * self.k * (q - self.x0)**2
        V12 = self.vc
        V22 = 0.5 * self.k * (q + self.x0)**2
        return np.array([[V11, V12], [V12, V22]], dtype=complex)

    def dH_dq(self, q: float) -> np.ndarray:
        dV11 = self.k * (q - self.x0)
        dV12 = 0.0
        dV22 = self.k * (q + self.x0)
        return np.array([[dV11, dV12], [dV12, dV22]], dtype=complex)


class SimpleAvoidedCrossing(DiabaticTwoState1D):
    """
    Tully-like 1D avoided crossing in a diabatic representation.
    Diabatic Hamiltonian:
        H_11(q) =  A (1 - tanh(B q))
        H_22(q) = -A (1 - tanh(B q))
        H_12(q) =  C exp(-D q^2)
    """

    def __init__(
        self, A: float = 0.01, B: float = 1.6, C: float = 0.005, D: float = 1.0, mass: float = 1.0
    ) -> None:
        super().__init__(mass=mass)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def H(self, q: float) -> np.ndarray:
        V11 = self.A * (1.0 - np.tanh(self.B * q))
        V22 = -self.A * (1.0 - np.tanh(self.B * q))
        V12 = self.C * np.exp(-self.D * q**2)
        return np.array([[V11, V12], [V12, V22]], dtype=float)

    def dH_dq(self, q: float) -> np.ndarray:
        Bq = self.B * q
        tanh_Bq = np.tanh(Bq)
        sech2 = 1.0 - tanh_Bq**2  # Stable 1/cosh^2

        dV11 = -self.A * self.B * sech2
        dV22 = -dV11
        V12 = self.C * np.exp(-self.D * q**2)
        dV12 = -2.0 * self.D * q * V12

        return np.array([[dV11, dV12], [dV12, dV22]], dtype=float)
    

def landau_traj(v: float, t: float, t0: float) -> float:
    """Landau-Zener classical trajectory: q(t) = v * (t - t0)."""
    return v * (t - t0)

class DoubleHarmonic3D(DiabaticTwoState3D):
    """
    3D version of 1D double harmonic well along the x axis. The other axes are simple harmonic.
    """
    def __init__(self, k: float, x0: float, vc: float, mass: float = 1.0) -> None:
        super().__init__(mass)
        self.k = k
        self.x0 = x0
        self.vc = vc

    def H(self, q: np.ndarray) -> np.ndarray:
        qx = q[0]
        qy = q[1]
        qz = q[2]
        V11 = 0.5 * self.k * (qx - self.x0)**2 + 1.0 * qy ** 2 + 1.0 * qz ** 2
        V12 = self.vc
        V22 = 0.5 * self.k * (qx + self.x0)**2 + 1.0 * qy ** 2 + 1.0 * qz ** 2
        return np.array([[V11, V12], [V12, V22]], dtype=complex)

    def dH_dq(self, q: np.ndarray) -> np.ndarray:
        qx = q[0]
        qy = q[1]
        qz = q[2]
        dV11x = self.k * (qx - self.x0)
        dV11y = 2.0 * qy
        dV11z = 2.0 * qz
        dV12 = 0.0
        dV22x = self.k * (qx + self.x0)
        dV22y = 2.0 * qy
        dV22z = 2.0 * qz
        
        dHx = np.array([[dV11x, dV12], [dV12, dV22x]])
        dHy = np.array([[dV11y, dV12], [dV12, dV22y]])
        dHz = np.array([[dV11z, dV12], [dV12, dV22z]])

        dH = np.array([dHx, dHy, dHz], dtype=complex)
        return dH

class LiftedGS3D(DiabaticTwoState3D):
    """
    Two harmonic oscillators in 3D that has vertically displaced x degree of freedom.
    """
    def __init__(self, k: float, x0: float, vc: float, disp: float, mass: float = 1.0) -> None:
        super().__init__(mass)
        self.k = k
        self.x0 = x0
        self.vc = vc
        self.disp = disp

    def H(self, q: np.ndarray) -> np.ndarray:
        qx = q[0]
        qy = q[1]
        qz = q[2]
        V11 = 0.5 * self.k * (qx - self.x0)**2 + self.disp + 1.0 * qy ** 2 + 1.0 * qz ** 2 # control the vertical displacement of the right parabola
        V12 = self.vc
        V22 = 0.5 * self.k * (qx + self.x0)**2 + 1.0 * qy ** 2 + 1.0 * qz ** 2
        return np.array([[V11, V12], [V12, V22]], dtype=complex)

    def dH_dq(self, q: np.ndarray) -> np.ndarray:
        qx = q[0]
        qy = q[1]
        qz = q[2]
        dV11x = self.k * (qx - self.x0)
        dV11y = 2.0 * qy
        dV11z = 2.0 * qz
        dV12 = 0.0
        dV22x = self.k * (qx + self.x0)
        dV22y = 2.0 * qy
        dV22z = 2.0 * qz
        
        dHx = np.array([[dV11x, dV12], [dV12, dV22x]])
        dHy = np.array([[dV11y, dV12], [dV12, dV22y]])
        dHz = np.array([[dV11z, dV12], [dV12, dV22z]])

        dH = np.array([dHx, dHy, dHz], dtype=complex)
        return dH
    

class MetastableExcited3D(DiabaticTwoState3D):
    '''
    Analytical model to simulate a metastable state in the upper state instead of the lower (ground) state.
    The ground state is a symmetric double well potential.
    '''
    def __init__(self, xmin_1, xmin_2, xbar, K, disp, A, xgs, vc, mass = 1):
        super().__init__(mass)
        self.xmin_1 = xmin_1
        self.xmin_2 = xmin_2
        self.xbar = xbar
        self.K = K
        self.disp = disp
        self.A = A
        self.xgs = xgs
        self.vc = vc
        
    
    def H(self, q: np.ndarray) -> np.ndarray:
        S2 = self.xmin_1 + self.xbar + self.xmin_2
        S1 = self.xmin_1 * self.xbar + self.xmin_1 * self.xmin_2 + self.xbar * self.xmin_2
        P = self.xmin_1 * self.xmin_2 * self.xbar
        qx = q[0]
        qy = q[1]
        qz = q[2]
        
        V11 = self.A * (qx ** 2 - self.xgs ** 2) ** 2 + qy ** 2 + qz ** 2
        V22 = self.K * (0.25 * qx ** 4 - S2/3 * qx ** 3 + 0.5 * S1 * qx ** 2 + P * qx) + self.disp + qy ** 2 + qz ** 2
        V12 = self.vc

        return np.array([[V11, V12], [V12, V22]], dtype=complex)
    
    def dH_dq(self, q):
        qx = q[0]
        qy = q[1]
        qz = q[2]

        dV11x = 4.0 * self.A * qx * (qx ** 2 - self.xgs ** 2)
        dV22x = self.K * (qx - self.xmin_1) * (qx - self.xmin_2) * (qx - self.xbar)
        dV11y = 2.0 * qy
        dV22y = 2.0 * qy
        dV11z = 2.0 * qz
        dV22z = 2.0 * qz
        dV12 = 0.0

        dHx = np.ndarray([[dV11x, dV12], [dV12, dV22x]])
        dHy = np.ndarray([[dV11y, dV12], [dV12, dV22y]])
        dHz = np.ndarray([[dV11z, dV12], [dV12, dV22z]])

        dH = np.array([dHx, dHy, dHz], dtype=complex)
        return dH


