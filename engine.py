from __future__ import annotations

from typing import Callable, List, Union

import numpy as np

from analytical import DiabaticTwoState1D
from integrator import (
    verlet_v,
    verlet_X,
    velocity_rescaling,
)
from trajectory import Snapshot, Trajectory
from electronic import sz_from_coeff, hop_search, local_diabatisation

class MASHEngine:
    """MASH propagation engine with reversible surface hops."""

    def __init__(
        self,
        model: DiabaticTwoState1D,
        dt: float,
        initial_snapshot: Snapshot,
    ) -> None:
        """
        Initialize engine.

        Args:
            model: Diabatic two-state model used for forces and eigensystem.
            dt: Nuclear time step.
            initial_snapshot: Initial nuclear and electronic state.
        """
        self.model = model
        self.dt = dt
        self.initial_snapshot = initial_snapshot
        self.mass = initial_snapshot.mass
    
    def _move_step(
        self,
        snapshot: Snapshot,
        timer: int,
    ) -> Union[Snapshot, List[Snapshot]]:
        """
        Propagate one MASH step, including possible hop.

        If a hop occurs within the step, returns a list of two snapshots:
        one at the hop and one at the end of the full time step.[web:59]

        Args:
            snapshot: Current snapshot.
            timer: Integer time index (for logging).

        Returns:
            Either a single next snapshot or a list [snapshot_at_hop, snapshot_end].
        """
        snapshot_next = self._unit_step(snapshot=snapshot, dt=self.dt, is_grid=True)

        Sz_cur = sz_from_coeff(snapshot.coefficients)
        Sz_next = sz_from_coeff(snapshot_next.coefficients)

        if Sz_cur * Sz_next < 0.0:
            tau_M = hop_search(
                dt=self.dt,
                snapshot=snapshot,
                model=self.model,
            )
            snapshot_1 = self._unit_step(
                snapshot=snapshot,
                dt=tau_M,
                is_grid=False,
            )

            v_new, is_hop = velocity_rescaling(
                model=self.model,
                snapshot=snapshot_1,
            )
            if is_hop:
                # print('Hop is allowed, at time', timer)
                active_state_new = 1 - snapshot_1.active_state
                snapshot_1.hop = True
            else:
                # print("hop is frustrated, at time", timer)
                active_state_new = snapshot_1.active_state
                snapshot_1.hop = False

            inter_snapshot = Snapshot(
                positions=snapshot_1.positions,
                velocities=v_new,
                coefficients=snapshot_1.coefficients,
                active_state=active_state_new,
                gauge=snapshot_1.gauge,
                mass=self.mass,
                is_grid=False,
            )
            
            dt_R = self.dt - tau_M
            snapshot_2 = self._unit_step(
                snapshot=inter_snapshot,
                dt=dt_R,
                is_grid=True,
            )
            return [snapshot_1, snapshot_2]

        return snapshot_next
    
    def propagate(self, n_steps: int) -> Trajectory:
        """
        Propagate for a fixed number of steps.

        Args:
            n_steps: Number of nuclear time steps.

        Returns:
            Trajectory containing all snapshots, including hops when present.
        """
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        for time_stamp in range(n_steps):
            snapshot_cur = traj[-1]
            snapshot_next = self._move_step(
                snapshot=snapshot_cur,
                timer=time_stamp,
            )

            if isinstance(snapshot_next, list):
                for s in snapshot_next:
                    traj.append(s)
            else:
                traj.append(snapshot_next)

        return traj
    
    def propagate_until_basin(
        self,
        max_steps: int,
        stateA: Callable[[float, int], bool],
        stateB: Callable[[float, int], bool],
    ) -> Trajectory:
        """
        Propagate until entering either of two basins or max_steps is reached.

        Basin predicates should take (position, active_state) and return True
        if the snapshot is in that basin.

        Args:
            max_steps: Maximum number of time steps to propagate.
            stateA: Basin A predicate.
            stateB: Basin B predicate.

        Returns:
            Trajectory up to (and including) the first basin entry.
        """
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        for time_stamp in range(max_steps):
            snapshot_cur = traj[-1]
            if stateA(snapshot_cur.positions, snapshot_cur.active_state) or stateB(
                snapshot_cur.positions,
                snapshot_cur.active_state,
            ):
                break

            snapshot_next = self._move_step(
                snapshot=snapshot_cur,
                timer=time_stamp,
            )

            if isinstance(snapshot_next, list):
                for s in snapshot_next:
                    traj.append(s)
            else:
                traj.append(snapshot_next)

        return traj
    
    def propagate_until_X(
        self,
        max_steps: int,
        stateA: Callable[[float, int], bool],
        stateB: Callable[[float, int], bool],
    ) -> Trajectory:
        """
        Propagate until the trajectory reaches the opposite basin.

        The seed must start in exactly one of stateA or stateB, and the
        trajectory is propagated until it first enters the *other* basin.

        Args:
            max_steps: Maximum number of time steps.
            stateA: Basin A predicate.
            stateB: Basin B predicate.

        Returns:
            Trajectory up to and including first hit of the target basin.

        Raises:
            ValueError: If the initial snapshot is in neither basin.
            RuntimeError: If the target basin is not reached in max_steps.
        """
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        start_A = stateA(
            self.initial_snapshot.positions,
            self.initial_snapshot.active_state,
        )
        start_B = stateB(
            self.initial_snapshot.positions,
            self.initial_snapshot.active_state,
        )

        if not (start_A or start_B):
            raise ValueError("Seed must start in a designated basin.")

        target_state = stateB if start_A else stateA

        for time_stamp in range(max_steps):
            snapshot_cur = traj[-1]
            if target_state(snapshot_cur.positions, snapshot_cur.active_state):
                return traj

            snapshot_next = self._move_step(
                snapshot=snapshot_cur,
                timer=time_stamp,
            )
            if isinstance(snapshot_next, list):
                for s in snapshot_next:
                    traj.append(s)
            else:
                traj.append(snapshot_next)

        raise RuntimeError(
            f"Seed trajectory failed to reach target basin within {max_steps} steps.",
        )

    def _unit_step(
        self,
        snapshot: Snapshot,
        dt: float,
        is_grid: bool,
    ) -> Snapshot:
        """
        Single nuclear-electronic propagation step without hop handling.

        Uses velocity Verlet for nuclei and local diabatization for the
        electronic coefficients.[web:60][web:59]

        Args:
            snapshot: Current snapshot.
            dt: Time step.
            is_grid: Whether the resulting snapshot lies on the regular grid.

        Returns:
            New Snapshot after one unit step.
        """
        q_cur = snapshot.positions
        v_cur = snapshot.velocities
        active_state = snapshot.active_state

        v_half = verlet_v(
            dt=dt,
            model=self.model,
            mass=self.mass,
            v_curr=v_cur,
            q_curr=q_cur,
            active_state=active_state,
        )
        q_next = verlet_X(dt=dt, v_half=v_half, q_curr=q_cur)
        v_next = verlet_v(
            dt=dt,
            model=self.model,
            mass=self.mass,
            v_curr=v_half,
            q_curr=q_next,
            active_state=active_state,
        )

        C_next, coeff_next = local_diabatisation(
            model=self.model,
            snapshot=snapshot,
            q_next=q_next,
            dt=dt,
        )
        return Snapshot(
            positions=q_next,
            velocities=v_next,
            coefficients=coeff_next,
            active_state=active_state,
            gauge=C_next,
            mass=self.mass,
            is_grid=is_grid,
        )
    
    
class MASHEngineIrrev:
    """MASH engine with irreversible hopping rule."""

    def __init__(
        self,
        model: DiabaticTwoState1D,
        dt: float,
        initial_snapshot: Snapshot,
    ) -> None:
        """
        Initialize irreversible MASH engine.

        Args:
            model: Diabatic two-state model used for propagation.
            dt: Nuclear time step.
            initial_snapshot: Initial nuclear and electronic state.
        """
        self.model = model
        self.dt = dt
        self.initial_snapshot = initial_snapshot
        self.mass = initial_snapshot.mass

    def _move_step(self, snapshot: Snapshot) -> Snapshot:
        """
        Propagate a single step and apply irreversible hopping rule.

        If a hop occurs and is allowed, switch the active state; if hop
        is frustrated, flip the coefficients instead.[web:59]

        Args:
            snapshot: Current snapshot.

        Returns:
            Next snapshot after the (possibly hopping) step.
        """
        snapshot_next = self._unit_step(snapshot=snapshot, dt=self.dt)

        Sz_cur = sz_from_coeff(snapshot.coefficients)
        Sz_next = sz_from_coeff(snapshot_next.coefficients)

        if Sz_cur * Sz_next < 0.0:
            v_new, is_hop = velocity_rescaling(
                model=self.model,
                snapshot=snapshot_next,
            )
            if is_hop:
                print("Hop is allowed")
                active_state_new = 1 - snapshot_next.active_state
                coeff_next_new = snapshot_next.coefficients
            else:
                print("Hop is frustrated")
                active_state_new = snapshot_next.active_state
                coeff_next_new = np.array(
                    [
                        snapshot_next.coefficients[1].conj(),
                        snapshot_next.coefficients[0].conj(),
                    ],
                    dtype=complex,
                )

            snapshot_next = Snapshot(
                positions=snapshot_next.positions,
                velocities=v_new,
                coefficients=coeff_next_new,
                active_state=active_state_new,
                mass=self.mass,
                gauge=snapshot_next.gauge,
                is_grid=snapshot_next.is_grid,
            )

        return snapshot_next

    def propagate(self, n_steps: int) -> Trajectory:
        """
        Propagate for a fixed number of steps with irreversible hops.

        Args:
            n_steps: Number of nuclear time steps.

        Returns:
            Trajectory with all snapshots.
        """
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        for _ in range(n_steps):
            snapshot_cur = traj[-1]
            snapshot_next = self._move_step(snapshot=snapshot_cur)

            if isinstance(snapshot_next, list):
                for s in snapshot_next:
                    traj.append(s)
            else:
                traj.append(snapshot_next)

        return traj

    def _unit_step(self, snapshot: Snapshot, dt: float) -> Snapshot:
        """
        Single nuclear-electronic step without hop logic (irreversible engine).

        Args:
            snapshot: Current snapshot.
            dt: Time step.

        Returns:
            New Snapshot after one unit step.
        """
        q_cur = snapshot.positions
        v_cur = snapshot.velocities
        active_state = snapshot.active_state

        v_half = verlet_v(
            dt=dt,
            model=self.model,
            mass=self.mass,
            v_curr=v_cur,
            q_curr=q_cur,
            active_state=active_state,
        )
        q_next = verlet_X(dt=dt, v_half=v_half, q_curr=q_cur)
        v_next = verlet_v(
            dt=dt,
            model=self.model,
            mass=self.mass,
            v_curr=v_half,
            q_curr=q_next,
            active_state=active_state,
        )

        C_next, coeff_next = local_diabatisation(
            model=self.model,
            snapshot=snapshot,
            q_next=q_next,
            dt=dt,
        )
        return Snapshot(
            positions=q_next,
            velocities=v_next,
            coefficients=coeff_next,
            active_state=active_state,
            gauge=C_next,
            mass=self.mass,
            is_grid=snapshot.is_grid,
            )
