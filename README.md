
Restrict the experimentation to two-state MASH.

## 1. Analytical models

A class that constructs pre-defined two-state Hamiltonian and provides all relevant information pertinent to it.
```
public type BaseModel
	
	initiate attributes:
		hamiltonian: H, dtype = complex, struct = (2,2)
		derivative: dH_dq, dtype = complex, struct = (2,2)
	
	derived attributes:
		adiabatic_energies: (V1(q), V2(q)), dtype = float, struct = scalar
		forces: (dV1(q), dV2(q)), dtype = float, struct = scalar
		nacdr: d10, dtype = float, struct = scalar
		diabatic_representation: [C1, C2], dtype = float, struct = scalar
```

The user must manually implement the form of the Hamiltonian and its position derivative based on the `BaseModel` class. Currently, there are two models available: Landau-Zener and coupled 1D oscillator.

Most attributes information is self-explanatory, with the `diabatic_representation` being the adiabatic Hamiltonian eigenstates expressed as the linear combination of diabatic basis set. Later in trajectory and electronic propagation, the effect of global phase will be addressed.

## 2. Trajectory and snapshots

Introduce the snapshot object in two-level MASH:
```
public type Snapshot

	initiate attributes:
		position: x, dtype = float, struct = scalar
		velocity: v, dtype = float, struct = scalar
		coefficients: [c1, c2], dtype = complex, struct = (2)
		phased_diabatic_representation: [C1, C2], dtype = complex, struct =(2,2)
		mass: m, dtype = float, struct = scalar
		active_state: 1/0, dtype = integer, struct = scalar
```

Trajectory is defined as a list of snapshots in the order of time. In addition, for better bookkeeping, include time step involved for a trajectory (To do).

In order to preserve microscopic reversibility, the history of the global phase in each step must be stored, in the form of phase-modified adiabatic basis expressed under the diabatic basis set, `phased_diabatic_representation`.

It is noted with care that `coefficients` is the wavefunction under the *adiabatic* basis.

## 3. Electronic propagation

In this implementation, only local diabatisation (LD) propagator is available. According to the original paper (J. Chem. Phys. 137, 22A514 (2012)), the algorithm is as follows:

```
function local_diabatisation

	in: TRAJ, MODEL, q_next
	include: align_phase, lowdin_orthonormalise, unitary_propagator
	out: C_next, coeff_next

	obtain position as q_curr from TRAJ
	obtain phased_diabatic_representation as C_curr from TRAJ
	obtain coefficients as coeff_curr from TRAJ
	compute C_next_raw at q_next from MODEL.diabatic_representation
	
	align_phase: C_next_raw -> C_next according to C_curr
	form E_curr and E_next ! diagonal adiabatic eigenvalues matrix
	form S_overlap from C_next.conj().T @ C_curr
	
	T = lowdin_orthonormalise(S_overlap)
	HLD = T @ E_next @ T.conj().T
	U = unitary_propagator(1/2 * (HLD + E_curr)) with timestep dt
	coeff_next = T.conj().T @ U @ coeff_curr
	
end function
```

The embedded functions `align_phase, lowdin_orthonormalise, unitary_propagator` are utility functions whose detailed working is not the main focus of this manual.

## 4. Piecewise-continuous event management

During a step where (one) hop happens, the `Sz` derived from `coefficients` will switch sign. Under this condition, a root search is triggered as follows:

```
! Finished nuclear propagation...
! Finished electronic propagation...

obtain coeff_curr from TRAJ
Sz_curr = abs(coeff_curr(1))**2 - abs(coeff_curr(0))**2
Sz_next = abs(coeff_next(1))**2 - abs(coeff_next(0))**2

if Sz_curr * Sz_next < 0: ! a hop is detected
	! call hop_search...
	! do modified propagation part I...
	! (at the hopping point) do velocity rescaling...
	! do modified propagation part II...
end if
```

The exact procedure of a hop search reads:

```
function hop_search
	in: dt, TRAJ, MODEL, tol_tau, tol_Sz, max_iter
	out: tau_M
	
	obtain coefficients as coeff_curr from TRAJ (last SNAPSHOT)
	obtain position as q_curr from TRAJ
	obtain velocity as v_curr from TRAJ
	obtain active_state from TRAJ
	
	tau_L = 0.0 ! The initial bracket of time interval
	tau_R = dt
	Sz_curr = abs(coeff_curr(1))**2 - abs(coeff_curr(0))**2
	iter = 0
	
	while True:
		tau_M = 0.5 * (tau_L + tau_R)
		v_half = verlet_v(v_curr) with timestep tau_M
		q_next = verlet_x(q_curr, v_half) with timestep tau_M
		C_next, coeff_next = local_diabatisation(TRAJ, MODEL, q_next)
		Sz_next = abs(coeff_next(1))**2 - abs(coeff_next(0))**2
		
		if abs(Sz_next) < tol_Sz .or. abs(tau_R - tau_L) < tol_tau:
			break while
			
		if Sz_curr * Sz_next < 0: ! if hop lands in the left bracket:
			tau_R = tau_M
		else:
			tau_L = tau_M
			
		iter = iter + 1
		if iter > max_iter:
			raise Error('Maximum iteration reached')
	end while
	
end function
```

After a suitable `tau_M` has been located, all that needs to be done is to propagate the system from 0 (beginning of the hopping step) to `tau_M`, do rescaling, and then from `tau_M` to 1 (just after the hopping step). The rescaling procedure is outlined as follows:

```
function velocity_rescaling
	in: MODEL, TRAJ
	out: v_new
	
	obtain mass from SNAPSHOT ! By convention, take from the last snapshot
	obtain active_state from SNAPSHOT
	obtain velocity as v_old from TRAJ
	obtain position as q_hop from TRAJ
	compute d10 from MODEL at q_hop
	compute adiabatic_energies from MODEL at q_hop
	
	V_init = adiabatic_energies(active_state)
	V_fin = adiabatic_energies(1 - active_state)
	
	mw_p_old = sqrt(mass) * v_old
	mw_nac = sqrt(mass) * d10
	
	mw_p_old_d = dot(mw_nac, mw_p_old) * 1/|mw_nac|
	
	E_d = 0.5 * mw_p_old_d**2 + V_init
	
	if E_d >= V_fin:
		mw_p_new_d = sign(mw_p_old_d) * sqrt(mw_p_old_d**2 + 2 * (V_init-V_fin))
		mw_p_new = mw_p_old - mw_nac/|mw_nac| * (mw_p_old_d - mw_p_new_d)
	else:
		mw_p_new = mw_p_old - 2 * mw_nac/|mw_nac| * mw_p_old_d
	end if
	
	v_new = mw_p_new/sqrt(mass)
	
	
end function
```

## 5. Nuclear integration

Nuclear integration consists simply of the velocity-Verlet position and velocity integration, which are the conventional algorithms. No new modifications are introduced, so they will simply be referred to as `verlet_v` and `verlet_x` in other contexts.

## 6. Standard workflow and bookkeeping

It is important to keep in mind when and to where the information obtained from a new step propagation should be stored because the existence of `hop_search` essentially makes any propagation effort before it a trial instead of a verdict. In other words, when to hop is never known a priori. The following workflow is proposed to ensure that the correct information is stored in the correct location.

```
program Dynamics

read in initial conditions: q, v, coeff, C, mass
read in initial parameter: hamiltonian parameters, dt, max_timestep

build SNAPSHOT, name = initial_snapshot
	position = q
	velocity = v
	coefficients = coeff
	phased_diabatic_representation = C
	mass = mass
	
initialise TRAJ, name = trajectory

initialise MODEL, name = model

append initial_snapshot(SNAPSHOT) into trajectory(TRAJ)

for step = 0, max_timestep:
	obtain position, velocity as q_curr, v_curr from trajectory[-1]
	obtain coefficients as coeff_curr from trajectory[-1]
	obtain mass from trajectory[-1]
	obtain active_state from trajectory[-1]
	v_half = verlet_v(v_curr) with timestep dt
	q_next = verlet_x(q_curr, v_half) with timestep dt
	v_next = verlet_v(v_half) with timestep dt
	
	C_next, coeff_next = local_diabatisation(trajectory, model)
	Sz_curr = abs(coeff_curr(1))**2 - abs(coeff_curr(0))**2
	Sz_next = abs(coeff_next(1))**2 - abs(coeff_next(0))**2

	if Sz_curr * Sz_next < 0:
		tau_M = hop_search(trajectory, model, dt)
		v_half_1 = verlet_v(v_curr) with timestep tau_M
		q_next_1 = verlet_x(q_curr, v_half_1) with timestep tau_M
		v_next_1 = verlet_v(v_half_1) with timestep tau_M
		
		C_next_1, coeff_next_1 = local_diabatisation(trajectory, model, q_next_1)
		
		build SNAPSHOT, name = last_snapshot
			position = q_next_1
			velocity = v_next_1
			coefficients = coeff_next_1
			phased_diabatic_representation = C_next_1
			mass = mass
		append last_snapshot(SNAPSHOT) into trajectory
		
		v_new = velocity_rescaling(model, trajectory)
		
		v_half_2 = verlet_v(v_new) with timestep dt - tau_M
		q_next_2 = verlet_x(q_next_1, v_half) with timestep dt - tau_M
		v_next_2 = verlet_v(v_half_2) with timestep dt - tau_M
		
		C_next_2, coeff_next_2 = local_diabatisation(trajectory, model, q_next_2)
		
		build SNAPSHOT, name = last_snapshot
			position = q_next_2
			velocity = v_next_2
			coefficients = coeff_next_2
			phased_diabatic_representation = C_next_2
			mass = mass
		append last_snapshot(SNAPSHOT) into trajectory
	else:
		build SNAPSHOT, name = last_snapshot
			position = q_next
			velocity = v_next
			coefficients = coeff_next
			phased_diabatic_representation = C_next
			mass = mass
		append last_snapshot(SNAPSHOT) into trajectory
	end if
	
end for

save trajectory, model

end program
```

A few points to notice:
1. During a split timestep due to hop search, two, instead of one, snapshots will be created and recorded.
2. The second snapshot in the split timestep is based on the rescaled velocity.
3. The phase memory is kept by extracting the diabatic `C` in the previous snapshot in each round of propagation.
4. `Sz` is not directly propagated in this setup, and nor are `Sx` and `Sy`. Here `Sz` only serves as an indicator of hop.
5. In principle, the switch of active state can be implicitly done by calculating based on the stored coefficient, but to prevent numerical edge cases at hopping, active state is manually set to correspond to `Sz`. If no hop happens, then the active state should be the same as that of the previous snapshot; if there is a hop, the state is switched.
