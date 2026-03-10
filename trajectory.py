import numpy as np
import copy
from typing import List, Optional, Union


class Snapshot:
    """Represents a single snapshot in a trajectory with positions, velocities, etc."""
    
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        coefficients: np.ndarray,
        active_state: int,
        gauge: np.ndarray,
        mass: float,
        is_grid: bool,
        hop=None
    ):
        self.positions = positions
        self.velocities = velocities
        self.coefficients = coefficients
        self.gauge = gauge
        self.active_state = active_state
        self.mass = mass
        self.is_grid = is_grid
        self.hop = hop
        self.shooting = False
    
    def reversed(self) -> Snapshot:
        """Return a reversed snapshot with negated velocities and conjugated coefficients."""
        reverse = copy.deepcopy(self)
        reverse.velocities *= -1.0
        reverse.coefficients = np.conjugate(reverse.coefficients)
        return reverse
    

class Trajectory:
    """Manages a sequence of Snapshot states."""

    def __init__(self, snaps: Optional[List[Snapshot]] = None):
        """Initialize with optional iterable of Snapshot states."""
        if snaps is None:
            self._snaps: List[Snapshot] = []
        else:
            self._snaps = list(snaps)

    def __len__(self) -> int:
        """Return the number of snapshots (enables len(traj))."""
        return len(self._snaps)

    def __getitem__(self, index: Union[int, slice]) -> Union[Snapshot, "Trajectory"]:
        """Access state by index or slice (enables traj[i])."""
        snap_slice = self._snaps[index]
        if isinstance(snap_slice, Snapshot):
            return snap_slice
        return Trajectory(snap_slice)

    def append(self, snap: Snapshot) -> None:
        """Add a new Snapshot to the trajectory."""
        self._snaps.append(snap)

    def __repr__(self) -> str:
        """String representation showing trajectory length."""
        return f"Trajectory(length={len(self)})"
