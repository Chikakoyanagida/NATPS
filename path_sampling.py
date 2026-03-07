from analytical import *
from engine import *
from selector import *
from trajectory import *
import copy
import numpy as np

class PathSampling:
    
    def __init__(self, selector, seed_traj, max_iter, max_length, in_stateA, in_stateB):
        self.seed_traj = seed_traj
        self.max_iter = max_iter
        self.max_length = max_length
        self.selector = selector
        self.in_stateA = in_stateA
        self.in_stateB = in_stateB

    def run(self, engine, model, dt, direction='both'):
        current_traj = self.seed_traj
        tps_ensemble = [current_traj]
        naive_tps_ensemble = [current_traj]
        selector = self.selector
        n_tps_iterations = self.max_iter
        max_traj_length = self.max_length
        inA = self.in_stateA
        inB = self.in_stateB

        self.accept = 0
        self.reject = 0

        for i in range(n_tps_iterations):
            print(f"--- TPS Iteration {i} ---")
            shooting_snap, l_old = selector.select_and_perturb(current_traj)
            shooting_snap.shooting = True
            engine_fwd = engine(model=model, dt=dt, initial_snapshot=shooting_snap)
            traj_fwd = engine_fwd.propagate_until_basin(max_traj_length, inA, inB)

            shooting_snap_rev = shooting_snap.reversed()
            engine_bwd = engine(model=model, dt=dt, initial_snapshot=shooting_snap_rev)
            traj_bwd_raw = engine_bwd.propagate_until_basin(max_traj_length, inA, inB)
            traj_bwd_fixed = [snap.reversed() for snap in reversed(traj_bwd_raw)]

            stitched_states = traj_bwd_fixed[:-1] + list(traj_fwd)
            trial_traj = Trajectory(stitched_states)

            start_q = trial_traj[0].positions
            start_s = trial_traj[0].active_state
            end_q = trial_traj[-1].positions
            end_s = trial_traj[-1].active_state

            connected = False
            if direction == 'both':
                if inA(start_q, start_s) and inB(end_q, end_s): connected = True
                if inB(start_q, start_s) and inA(end_q, end_s): connected = True
            else:
                raise NotImplementedError('Directional path ensemble not implemented')
            
            naive_tps_ensemble.append(trial_traj)

            if selector.check_acceptance(l_old, trial_traj, connected):
                current_traj = trial_traj
                print('Accepted')
                self.accept += 1
            else:
                print('Rejected')
                self.reject += 1
                # shooting_snap.shooting = False # Not necessary
            
            tps_ensemble.append(current_traj)
        
        print(f'TPS finished! Ensemble size: {len(tps_ensemble)}')
        return tps_ensemble, naive_tps_ensemble
    
    def MC_acceptance(self,):
        MC_rate = self.accept/(self.accept + self.reject)
        print(f'MC acceptance rate: {MC_rate * 100}%')
        succ_traj = MC_rate * self.max_iter
        print(f'Accounts to {succ_traj} successfully sampled trajectories')
        return MC_rate
    
    def transit_time_statistics(self, tps_ensemble, dt, burn_in=0.15):
        burn_in_frames = int(burn_in * len(tps_ensemble))
        production_ensemble = tps_ensemble[burn_in_frames:]
        production_ensemble = [self._remove_off_grid(traj) for traj in production_ensemble]
        
        transit_times = [(len(traj)-1) * dt for traj in production_ensemble]
        mean_time = np.mean(transit_times)
        std_time = np.std(transit_times)

        print(f"Mean Transition Time: {mean_time:.2f} atomic units")
        print(f"Standard Deviation: {std_time:.2f} atomic units")

        return transit_times, mean_time, std_time
    
    def efficiency_statistics(self, naive_tps_ensemble, mc_rate, burn_in=0.15):
        burn_in_frames = int(burn_in * len(naive_tps_ensemble))
        production_ensemble = naive_tps_ensemble[burn_in_frames:]
        production_ensemble = [self._remove_off_grid(traj) for traj in production_ensemble]
        total_traj = len(production_ensemble)
        
        succ_traj = mc_rate * total_traj
        total_steps = sum([len(traj) for traj in production_ensemble])
        if succ_traj == 0:
            print("No successful trajectories. Efficiency approaches 0.")
            return float('inf')
        speedup = total_steps/succ_traj
        print(f'Mean number of steps run to find one transition path: {speedup}')

        return speedup
    
    def compute_autocorrelation(self, data, max_lag=100):
        
        mean = np.mean(data)
        var = np.var(data)
        data_centered = np.array(data) - mean
        
        N = len(data)
        autocorr = []
    
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                # Calculate covariance at the given lag
                cov = np.sum(data_centered[:-lag] * data_centered[lag:]) / (N - lag)
                autocorr.append(cov / var)
            
        return autocorr
    
    def hop_statistics(self, tps_ensemble, burn_in=0.15):
        burn_in_frames = int(burn_in * len(tps_ensemble))
        production_ensemble = tps_ensemble[burn_in_frames:]

        hop_counts = [sum(1 for snap in traj if getattr(snap, 'hop', False) is True) for traj in production_ensemble]
        sum_hops = np.sum(hop_counts)
        mean_hops = np.mean(hop_counts)
        std_hops = np.std(hop_counts)
        print('total number of hops:', sum_hops)
        print(f"Mean Number of Hops: {mean_hops:.3f}")
        print(f"Standard Deviation of Hops: {std_hops:.3f}")

        q_hops = np.array([snap.positions
                           for traj in production_ensemble
                           for snap in traj if getattr(snap, 'hop', False) is True])

        return mean_hops, std_hops, q_hops
    
    @staticmethod
    def _remove_off_grid(traj):
        return Trajectory([snap for snap in traj if snap.is_grid])