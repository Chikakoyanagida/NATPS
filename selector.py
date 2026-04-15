import numpy as np
import copy

from typing import Callable, Iterable, Tuple, Optional

from electronic import *
from trajectory import Snapshot

class UniformShootingSelector:
    """
    Uniform shooting move selector for trajectory sampling with basin constraints.

    Grid snapshots that are inside either basin A or B are excluded; shooting
    points are drawn uniformly from the remaining frames and their velocities
    are perturbed with a Maxwell–Boltzmann draw.
    """

    def __init__(
        self,
        temp: float,
        mass: float,
        stateA_func: Callable[[float, int], bool],
        stateB_func: Callable[[float, int], bool],
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        alpha: float = 0.5,
    ) -> None:
        """
        Initialize selector.

        Args:
            temp: Temperature in Kelvin.
            mass: Mass in atomic units for the 1D degree of freedom.
            stateA_func: Predicate returning True if (position, active_state)
                is in basin A.
            stateB_func: Predicate returning True if (position, active_state)
                is in basin B.
            alpha: Mixing factor between old velocity and Maxwell–Boltzmann
                draw in [0, 1]. alpha=1 keeps v_old; alpha=0 uses v_MB only.
        """
        kB_au = 3.1668114e-6  # Boltzmann constant in Hartree/K (atomic units).[web:93]
        self.kbT = kB_au * temp
        self.mass = mass
        self.alpha = alpha
        self.inA = stateA_func
        self.inB = stateB_func

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

    def _is_eligible(self, snap: Snapshot) -> bool:
        """Return True if snapshot is on the grid and outside both basins A and B."""
        if not snap.is_grid:
            return False

        in_A = self.inA(snap.positions, snap.active_state)
        in_B = self.inB(snap.positions, snap.active_state)

        return not (in_A or in_B)

    def select_and_perturb(
        self,
        traj: Iterable[Snapshot],
    ) -> Tuple[Snapshot, int]:
        """
        Select a shooting snapshot from eligible frames and perturb its velocity.

        Eligibility requires `is_grid == True` and being outside both basins
        A and B. The velocity is updated via
            v_new = alpha * v_old + sqrt(1 - alpha^2) * v_MB,
        where v_MB is sampled from N(0, k_B T / m), i.e. a 1D Maxwell–Boltzmann
        component.

        Args:
            traj: Iterable of Snapshot objects.

        Returns:
            Tuple (shooting_snapshot, l_old):
                shooting_snapshot: Deep-copied and perturbed Snapshot.
                l_old: Number of eligible snapshots in the original trajectory.

        Raises:
            ValueError: If there is no eligible snapshot to shoot from.
        """
        eligible_snaps = [snap for snap in traj if self._is_eligible(snap)]
        l_old = len(eligible_snaps)
        if l_old <= 0:
            raise ValueError("Trajectory too short to shoot after padding!")

        shooting_index = self.rng.integers(0, l_old)
        shooting_snapshot = copy.deepcopy(eligible_snaps[shooting_index])

        v_old = shooting_snapshot.velocities
        v_mb = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.kbT / self.mass),
        )
        v_new = self.alpha * v_old + np.sqrt(1.0 - self.alpha**2) * v_mb

        shooting_snapshot.velocities = v_new

        return shooting_snapshot, l_old
    
    def select_and_perturb_EC(
            
    ):
        pass

    def check_acceptance(
        self,
        l_old: int,
        trial_traj: Iterable[Snapshot],
        connected: bool,
    ) -> bool:
        """
        Decide whether to accept a new trajectory in the uniform shooting scheme.

        If `connected` is False, the move is rejected outright. Otherwise,
        the number of eligible snapshots `l_new` in the trial trajectory is
        counted and the acceptance probability is
            P_acc = min(1, l_old / l_new).

        Args:
            l_old: Number of eligible snapshots in the reference trajectory.
            trial_traj: Trial trajectory (iterable of Snapshot).
            connected: Whether the trial trajectory is dynamically connected.

        Returns:
            True if the move is accepted, False otherwise.
        """
        if not connected:
            return False

        l_new = sum(1 for snap in trial_traj if self._is_eligible(snap))
        if l_new == 0:
            return False

        P_acc = min(1.0, l_old / l_new)
        rand_val = self.rng.uniform(0.0, 1.0)
        return bool(rand_val < P_acc)