import numpy as np

HBAR = 1.0

class DiabaticTwoState1D:
    """
    Base class for a 1D two-state system defined in a diabatic basis.

    You must implement:
        H(q)     -> 2x2 diabatic Hamiltonian matrix
        dH_dq(q) -> 2x2 derivative of Hamiltonian wrt q

    This class then provides:
        - adiabatic energies and NACs
        - forces on adiabatic surfaces
        - interface required by the MASH integrator
    """

    def __init__(self, mass=1.0):
        self.mass = float(mass)

    # ---- user must override these two ----
    def H(self, q: float) -> np.ndarray:
        raise NotImplementedError

    def dH_dq(self, q: float) -> np.ndarray:
        raise NotImplementedError

    # ---- internal helper: adiabatic info at q ----
    def adiabatic_info(self, q: float):
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

    # ---- API used by the integrator ----

    def V0(self, q: float) -> float:
        V0, V1, _, _, _, _ = self.adiabatic_info(q)
        return V0

    def V1(self, q: float) -> float:
        V0, V1, _, _, _, _ = self.adiabatic_info(q)
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
        """
        Force on adiabatic state a (0 or 1):
            F_a(q) = -dV_a/dq
        """
        V0, V1, dV0_dq, dV1_dq, _, _ = self.adiabatic_info(q)
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
    
    def eigvecs(self, q):
        _, _, _, _, _, eigvecs = self.adiabatic_info(q)
        return eigvecs
    

class LandauZener(DiabaticTwoState1D):
    """
    1D avoided crossing from Landau-Zener model (two linear diabatic surfaces)
    """
    def __init__(self, s, vc, mass=1):
        super().__init__(mass)
        self.s = s
        self.vc = vc
    
    def H(self, q):
        V11 = 0.0
        V12 = self.vc
        V22 = self.s * q
        return np.array([[V11, V12], [V12, V22]], dtype=float)
    
    def dH_dq(self, q):
        dV11 = 0.0
        dV12 = 0.0
        dV22 = self.s
        return np.array([[dV11, dV12], [dV12, dV22]], dtype=float)


class DoubleHarmonic(DiabaticTwoState1D):
    '''
    1D avoided crossing from two horizontally displaced simple harmonic oscillators.
    '''

    def __init__(self, k, x0, vc, mass=1):
        super().__init__(mass)
        self.k = k
        self.x0 = x0
        self.vc = vc
    
    def H(self, q):
        V11 = 1/2 * self.k * (q-self.x0)**2
        V12 = self.vc
        V22 = 1/2 * self.k * (q+self.x0)**2
        return np.array([[V11, V12], [V12, V22]], dtype=complex)
    
    def dH_dq(self, q):
        dV11 = self.k * (q-self.x0)
        dV22 = self.k * (q+self.x0)
        dV12 = 0.0
        return np.array([[dV11, dV12], [dV12, dV22]])


class SimpleAvoidedCrossing(DiabaticTwoState1D):
    """
    Tully-like 1D avoided crossing in a diabatic representation.
    Diabatic Hamiltonian:
        H_11(q) =  A (1 - tanh(B q))
        H_22(q) = -A (1 - tanh(B q))
        H_12(q) =  C exp(-D q^2)
    """

    def __init__(self, A=0.01, B=1.6, C=0.005, D=1.0, mass=1.0):
        super().__init__(mass=mass)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def H(self, q: float) -> np.ndarray:
        V11 = self.A * (1.0 - np.tanh(self.B * q))
        V22 = -self.A * (1.0 - np.tanh(self.B * q))
        V12 = self.C * np.exp(-self.D * q**2)
        return np.array([[V11, V12],
                         [V12, V22]], dtype=float)

    def dH_dq(self, q):
        # Use sech^2 = 1 - tanh^2 to avoid cosh overflow
        Bq = self.B * q
        tanh_Bq = np.tanh(Bq)
        sech2 = 1.0 - tanh_Bq**2  # = 1 / cosh^2(Bq) but numerically stable

        dV11 = -self.A * self.B * sech2
        dV22 = -dV11

        V12 = self.C * np.exp(-self.D * q**2)
        dV12 = -2.0 * self.D * q * V12

        return np.array([[dV11, dV12],
                        [dV12, dV22]], dtype=float)
    
def Landau_traj(v, t, t0):
    q = v * (t-t0)
    return q