import numpy as np
import copy
from electronic import *

class UniformShootingSelector:
    """
    Uniform shooting move selector for trajectory sampling.

    Chooses a shooting point from the grid snapshots, perturbs the velocity
    with a Maxwell–Boltzmann draw, and provides a Metropolis-like acceptance
    probability based on old/new effective lengths.
    """

    def __init__(self, temp: float, mass: float, alpha: float = 0.5) -> None:
        """
        Initialize selector with temperature, mass, and mixing parameter.

        Args:
            temp: Temperature in Kelvin.
            mass: Mass in atomic units for the 1D degree of freedom.
            alpha: Mixing factor between old velocity and new Maxwellian
                draw, in [0, 1]. alpha=1 keeps v_old, alpha=0 uses v_MB only.
        """
        kB_au = 3.1668114e-6  # Boltzmann constant in Hartree/K (atomic units).[web:93]
        self.kbT = kB_au * temp
        self.mass = mass
        self.alpha = alpha
    
    def select_and_perturb(
        self,
        traj,
        pad_L: int = 1,
        pad_R: int = 1,
    ) -> Tuple[Snapshot, int]:
        """
        Select a shooting point from grid snapshots and perturb its velocity.

        The shooting index is chosen uniformly from the interior portion of the
        grid snapshots [pad_L, L - pad_R), then the velocity is updated via
            v_new = alpha * v_old + sqrt(1 - alpha^2) * v_MB,
        with v_MB drawn from a 1D Maxwell–Boltzmann distribution
        N(0, k_B T / m).

        Args:
            traj: Iterable of Snapshot objects, some marked with is_grid=True.
            pad_L: Number of grid snapshots excluded at the left edge.
            pad_R: Number of grid snapshots excluded at the right edge.

        Returns:
            Tuple (shooting_snapshot, l_old):
                shooting_snapshot: Deep-copied and perturbed Snapshot.
                l_old: Effective length L_old - pad_L - pad_R used in acceptance.

        Raises:
            ValueError: If the trajectory is too short after padding.
        """
        grid_traj = [snap for snap in traj if snap.is_grid]
        L_old = len(grid_traj)
        l_old = L_old - pad_L - pad_R
        if l_old <= 0:
            raise ValueError("Trajectory too short to shoot after padding!")

        shooting_index = np.random.randint(pad_L, L_old - pad_R)
        shooting_snapshot = copy.deepcopy(grid_traj[shooting_index])

        v_old = shooting_snapshot.velocities
        v_mb = np.random.normal(
            loc=0.0,
            scale=np.sqrt(self.kbT / self.mass),
        )
        v_new = self.alpha * v_old + np.sqrt(1.0 - self.alpha**2) * v_mb

        shooting_snapshot.velocities = v_new

        return shooting_snapshot, l_old
    
    def check_acceptance(
        self,
        l_old: int,
        l_new: int,
        connected: bool,
    ) -> bool:
        """
        Decide whether to accept a new trajectory in uniform-shooting scheme.

        If `connected` is False, the move is rejected. Otherwise, acceptance
        probability is P_acc = min(1, l_old / l_new); a uniform random number
        in [0, 1) is compared against P_acc.

        Args:
            l_old: Old effective length (e.g., number of usable grid frames).
            l_new: New effective length.
            connected: Whether the new trajectory is dynamically connected.

        Returns:
            True if the move is accepted, False otherwise.
        """
        if not connected:
            return False

        P_acc = min(1.0, l_old / l_new)
        rand_val = np.random.uniform(0.0, 1.0)
        return bool(rand_val < P_acc)
    
