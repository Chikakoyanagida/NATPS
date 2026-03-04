from analytical import *
from electronic import *
from integrator import *
from trajectory import *
import numpy as np

class MASHEngine:
    
    def __init__(self, model, dt, initial_snapshot):
        self.model = model
        self.dt = dt
        self.initial_snapshot = initial_snapshot
        self.mass = initial_snapshot.mass
    
    def _move_step(self, snapshot, timer):
        snapshot_next = self._unit_step(snapshot=snapshot, dt=self.dt, is_grid=True)
        
        Sz_cur = sz_from_coeff(snapshot.coefficients)
        Sz_next = sz_from_coeff(snapshot_next.coefficients)

        if Sz_cur * Sz_next < 0:
            tau_M = hop_search(dt=self.dt,
                            snapshot=snapshot,
                            model=self.model)
            snapshot_1 = self._unit_step(snapshot=snapshot, dt=tau_M, is_grid=False)

            v_new, is_hop = velocity_rescaling(model=self.model, snapshot=snapshot_1)
            if is_hop:
                print('Hop is allowed, at time', timer)
                active_state_new = 1 - snapshot_1.active_state
            else:
                print('hop is frustrated, at time', timer)
                active_state_new = snapshot_1.active_state
            
            inter_snapshot = Snapshot(positions=snapshot_1.positions,
                                      velocities=v_new,
                                      coefficients=snapshot_1.coefficients,
                                      active_state=active_state_new,
                                      gauge=snapshot_1.gauge,
                                      mass=self.mass,
                                      is_grid=False)

            dt_R = self.dt - tau_M
            snapshot_2 = self._unit_step(inter_snapshot, dt=dt_R, is_grid=True)
            return [snapshot_1, snapshot_2]
        else:
            return snapshot_next
    
    def propagate(self, n_steps):
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        for time_stamp in range(n_steps):
            snapshot_cur = traj[-1]
            snapshot_next = self._move_step(snapshot=snapshot_cur, timer=time_stamp)
            
            if isinstance(snapshot_next, list):
                for s in snapshot_next: traj.append(s)
            else:
                traj.append(snapshot_next)
        
        return traj
    
    def propagate_until_basin(self, max_steps, stateA, stateB):
        traj = Trajectory()
        traj.append(self.initial_snapshot)
        
        for time_stamp in range(max_steps):
            snapshot_cur = traj[-1]
            if stateA(snapshot_cur.positions, snapshot_cur.active_state) or stateB(snapshot_cur.positions, snapshot_cur.active_state):
                break
            snapshot_next = self._move_step(snapshot=snapshot_cur, timer=time_stamp)

            if isinstance(snapshot_next, list):
                for s in snapshot_next: traj.append(s)
            else:
                traj.append(snapshot_next)
        
        return traj
    
    def propagate_until_X(self, max_steps, stateA, stateB):
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        start_A = stateA(self.initial_snapshot.positions, self.initial_snapshot.active_state)
        start_B = stateB(self.initial_snapshot.positions, self.initial_snapshot.active_state)

        if not (start_A or start_B):
            raise ValueError('Seed must start in a designated basin.')
            
        target_state = stateB if start_A else stateA

        for time_stamp in range(max_steps):
            snapshot_cur = traj[-1]
            if target_state(snapshot_cur.positions, snapshot_cur.active_state):
                return traj # Successfully hit the target basin
                
            snapshot_next = self._move_step(snapshot=snapshot_cur, timer=time_stamp)
            if isinstance(snapshot_next, list):
                for s in snapshot_next: traj.append(s)
            else:
                traj.append(snapshot_next)
        
        raise RuntimeError(f"Seed trajectory failed to reach target basin within {max_steps} steps.")

    def _unit_step(self, snapshot, dt, is_grid):
        q_cur = snapshot.positions
        v_cur = snapshot.velocities
        active_state = snapshot.active_state

        v_half = verlet_v(dt=dt, model=self.model, mass=self.mass, v_curr=v_cur, q_curr=q_cur,
                      active_state=active_state)
        q_next = verlet_X(dt=dt, v_half=v_half, q_curr=q_cur)
        v_next = verlet_v(dt=dt, model=self.model, mass=self.mass, v_curr=v_half, q_curr=q_next
                      , active_state=active_state)
        
        C_next, coeff_next = local_diabatisation(model=self.model,
                                             snapshot=snapshot,
                                             q_next=q_next,
                                             dt=dt)
        return Snapshot(positions=q_next,
                            velocities=v_next,
                            coefficients=coeff_next,
                            active_state=active_state,
                            gauge=C_next,
                            mass=self.mass,
                            is_grid=is_grid)
    
    
class MASHEngineIrrev:
    
    def __init__(self, model, dt, initial_snapshot):
        self.model = model
        self.dt = dt
        self.initial_snapshot = initial_snapshot
        self.mass = initial_snapshot.mass
    
    def _move_step(self, snapshot):
        snapshot_next = self._unit_step(snapshot=snapshot, dt=self.dt)
        
        Sz_cur = sz_from_coeff(snapshot.coefficients)
        Sz_next = sz_from_coeff(snapshot_next.coefficients)

        if Sz_cur * Sz_next < 0:
            v_new, is_hop = velocity_rescaling(model=self.model, snapshot=snapshot_next)
            if is_hop:
                print('Hop is allowed')
                active_state_new = 1 - snapshot_next.active_state
                coeff_next_new = snapshot_next.coefficients
            else:
                print('Hop is frustrated')
                active_state_new = snapshot_next.active_state
                coeff_next_new = [snapshot_next.coefficients[1].conj(),
                                  snapshot_next.coefficients[0].conj()]
            
            snapshot_next = Snapshot(positions=snapshot_next.positions,
                                     velocities=v_new,
                                     coefficients=coeff_next_new,
                                     active_state=active_state_new,
                                     mass=self.mass,
                                     gauge=snapshot_next.gauge)
        
        return snapshot_next
    
    def propagate(self, n_steps):
        traj = Trajectory()
        traj.append(self.initial_snapshot)

        for _ in range(n_steps):
            snapshot_cur = traj[-1]
            snapshot_next = self._move_step(snapshot=snapshot_cur)
            
            if isinstance(snapshot_next, list):
                for s in snapshot_next: traj.append(s)
            else:
                traj.append(snapshot_next)
        
        return traj

    def _unit_step(self, snapshot, dt):
        q_cur = snapshot.positions
        v_cur = snapshot.velocities
        active_state = snapshot.active_state

        v_half = verlet_v(dt=dt, model=self.model, mass=self.mass, v_curr=v_cur, q_curr=q_cur,
                      active_state=active_state)
        q_next = verlet_X(dt=dt, v_half=v_half, q_curr=q_cur)
        v_next = verlet_v(dt=dt, model=self.model, mass=self.mass, v_curr=v_half, q_curr=q_next
                      , active_state=active_state)
        
        C_next, coeff_next = local_diabatisation(model=self.model,
                                             snapshot=snapshot,
                                             q_next=q_next,
                                             dt=dt)
        return Snapshot(positions=q_next,
                            velocities=v_next,
                            coefficients=coeff_next,
                            active_state=active_state,
                            gauge=C_next,
                            mass=self.mass)