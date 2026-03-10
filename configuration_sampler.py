import numpy as np
import matplotlib.pyplot as plt

def effective_potential(q, epsilon, Vc):
    """The position-dependent part of the user's analytical distribution."""
    term1 = epsilon * (q**2 + 1.0)
    term2 = - epsilon * np.sqrt(4.0 * q**2 + (Vc / epsilon)**2)
    return term1 + term2

def get_energy(q, p, m, epsilon, Vc):
    kE = 0.5 * p**2/m
    PE = effective_potential(q, epsilon, Vc)
    return kE + PE

def generate_mash_initial_conditions(num_samples, mass, temp, epsilon, Vc):
    """
    Generates q, p, and S arrays for brute-force MASH dynamics.
    """
    kB_au = 3.1668114e-6
    beta = 1.0 / (kB_au * temp)
    
    # ---------------------------------------------------------
    # 1. Sample Momentum (p) - Standard Gaussian
    # variance = mass / beta
    # ---------------------------------------------------------
    p_std = np.sqrt(mass / beta)
    p_samples = np.random.normal(loc=0.0, scale=p_std, size=num_samples)
    
    # ---------------------------------------------------------
    # 2. Sample Spin (S) - Lower Hemisphere Uniform
    # ---------------------------------------------------------
    # Sz is strictly between -1 and 0 for the ground adiabatic state
    Sz_samples = np.random.uniform(-1.0, 0.0, size=num_samples)
    
    # Random angle around the z-axis
    theta_samples = np.random.uniform(0.0, 2.0 * np.pi, size=num_samples)
    
    # Compute Sx and Sy using spherical trigonometry constraints
    xy_radius = np.sqrt(1.0 - Sz_samples**2)
    Sx_samples = xy_radius * np.cos(theta_samples)
    Sy_samples = xy_radius * np.sin(theta_samples)
    
    # Pack into an array of shape (num_samples, 3)
    S_samples = np.column_stack((Sx_samples, Sy_samples, Sz_samples))
    
    # ---------------------------------------------------------
    # 3. Sample Position (q) - Metropolis-Hastings MCMC
    # ---------------------------------------------------------
    q_samples = np.zeros(num_samples)
    
    # MCMC parameters
    current_q = 0.0  # Start at the crossing seam
    current_V = effective_potential(current_q, epsilon, Vc)
    step_size = 0.5  # Adjust this if acceptance rate is too high/low
    
    # Burn-in phase (let the walker forget the starting point)
    for _ in range(2000):
        trial_q = current_q + np.random.normal(0, step_size)
        trial_V = effective_potential(trial_q, epsilon, Vc)
        
        # Metropolis acceptance criterion
        delta_V = trial_V - current_V
        if delta_V < 0 or np.random.rand() < np.exp(-beta * delta_V):
            current_q = trial_q
            current_V = trial_V
            
    # Production phase (sample with a lag to avoid autocorrelation)
    lag = 10
    accepted_moves = 0
    total_moves = 0
    
    for i in range(num_samples):
        for _ in range(lag):
            total_moves += 1
            trial_q = current_q + np.random.normal(0, step_size)
            trial_V = effective_potential(trial_q, epsilon, Vc)
            
            delta_V = trial_V - current_V
            if delta_V < 0 or np.random.rand() < np.exp(-beta * delta_V):
                current_q = trial_q
                current_V = trial_V
                accepted_moves += 1
                
        q_samples[i] = current_q
        
    print(f"MCMC Acceptance Rate for q: {accepted_moves/total_moves:.2%}")
    
    return q_samples, p_samples, S_samples


def spin_to_electronic_coefficients(S_samples):
    """
    Converts a shape (N, 3) array of spin vectors (Sx, Sy, Sz) 
    back into complex electronic coefficients C0 and C1.
    Assumes vectors are in the lower hemisphere (Sz <= 0).
    """
    # Extract components for readability
    Sx = S_samples[:, 0]
    Sy = S_samples[:, 1]
    Sz = S_samples[:, 2]
    
    # 1. Calculate C0 (forced to be purely real and positive)
    # The clip function ensures we don't accidentally get tiny negative 
    # numbers inside the sqrt due to floating point rounding errors.
    C0_magnitude_squared = np.clip((1.0 - Sz) / 2.0, 0.0, 1.0)
    C0 = np.sqrt(C0_magnitude_squared)
    
    # C0 is purely real, so we can cast it to a complex array
    C0_complex = C0 + 0.0j
    
    # 2. Calculate C1
    # We use the relationship: C1 = (Sx - i*Sy) / (2 * C0)
    C1_complex = (Sx - 1j * Sy) / (2.0 * C0)
    
    # Pack them together into a shape (N, 2) complex array
    # Column 0 is C0, Column 1 is C1
    coefficients = np.column_stack((C0_complex, C1_complex))
    
    return coefficients

