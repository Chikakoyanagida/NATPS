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
    
    def _move_step(self, snapshot):
        snapshot_next = self._unit_step(snapshot=snapshot, dt=self.dt)
        
        Sz_cur = sz_from_coeff(snapshot.coefficients)
        Sz_next = sz_from_coeff(snapshot_next.coefficients)

        if Sz_cur * Sz_next < 0:
            tau_M = hop_search(dt=self.dt,
                            snapshot=snapshot,
                            model=self.model)
            snapshot_1 = self._unit_step(snapshot=snapshot, dt=tau_M)

            v_new, is_hop = velocity_rescaling(model=self.model, snapshot=snapshot_1)
            if is_hop:
                active_state_new = 1 - snapshot_1.active_state
            else:
                active_state_new = snapshot_1.active_state
            
            inter_snapshot = Snapshot(positions=snapshot_1.positions,
                                      velocities=v_new,
                                      coefficients=snapshot_1.coefficients,
                                      active_state=active_state_new,
                                      gauge=snapshot_1.gauge,
                                      mass=self.mass)

            dt_R = self.dt - tau_M
            snapshot_2 = self._unit_step(inter_snapshot, dt=dt_R)
            return [snapshot_1, snapshot_2]
        else:
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
                active_state_new = 1 - snapshot_next.active_state
                coeff_next_new = snapshot_next.coefficients
            else:
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