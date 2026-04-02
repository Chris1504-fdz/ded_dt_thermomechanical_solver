import cupy as cp
import numpy as np
import zarr
import os
import gc
import cupyx.scipy.sparse as cusparse
from cupyx.scipy.sparse import linalg as cusparse_linalg
from cupyx.scipy.sparse.linalg import LinearOperator

# Import Core Modules
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
from gamma.simulator.func import elastic_stiff_matrix, constitutive_problem, disp_match

class DEDSimulator:
    """
    Modular, memory-optimized Thermo-Mechanical Simulator for DED Additive Manufacturing.
    Matrix-Free Assembly with Original Unaltered Material Physics.
    """
    def __init__(self, input_dir="../0_properties", geom_file='wall.k', output_path="stress_history.zarr"):
        self.input_dir = os.path.abspath(input_dir)
        self.geom_file = geom_file
        self.output_path = output_path
        
        # Physics Constants
        self.poisson = 0.3
        self.a1 = 10000
        self.tol = 1.0e-8
        self.Maxit = 20
        self.n_q = 8
        
        # State Variables
        self.domain = None
        self.heat_solver = None
        self.z_root = None
        
    def _load_materials(self):
        """Loads and caches temperature-dependent material properties."""
        path = lambda name: os.path.join(self.input_dir, f'materials/{name}')
        
        self.young1 = cp.array(np.loadtxt(path('IN718_Young_Debroy.txt'))[:,1] / 1e6)
        self.temp_young1 = cp.array(np.loadtxt(path('IN718_Young_Debroy.txt'))[:,0])
        
        self.Y1 = cp.array(np.loadtxt(path('IN718_Yield_Debroy.txt'))[:,1] / 1e6 * np.sqrt(2/3))
        self.temp_Y1 = cp.array(np.loadtxt(path('IN718_Yield_Debroy.txt'))[:,0])
        
        self.scl1 = cp.array(np.loadtxt(path('IN718_Alpha_Debroy.txt'))[:,1])
        self.temp_scl1 = cp.array(np.loadtxt(path('IN718_Alpha_Debroy.txt'))[:,0])

    def _init_state_arrays(self):
        """Initializes all CuPy physical state tensors."""
        n_e, n_n = self.domain.nE, self.domain.nN
        
        self.E = cp.zeros((n_e, self.n_q, 6), dtype=cp.float64)
        self.S = cp.zeros((n_e, self.n_q, 6), dtype=cp.float64)
        self.Ep_prev = cp.zeros((n_e, self.n_q, 6), dtype=cp.float64)
        self.Hard_prev = cp.zeros((n_e, self.n_q, 6), dtype=cp.float64)
        self.U = cp.zeros((n_n, 3), dtype=cp.float64)
        self.alpha_Th = cp.zeros((n_e, self.n_q, 6), dtype=cp.float64)
        
        # --- OPTIMIZED ZARR STORAGE ---
        self.z_root = zarr.open(self.output_path, mode='w')
        
        # Calculate worst-case maximum steps (using the smallest dt) to pre-allocate
        min_dt = 0.02
        self.max_expected_steps = int(self.domain.end_sim_time / min_dt) + 500
        
        # Pre-allocate full shapes on disk. No appending later.
        self.z_stress = self.z_root.create_array('stress', shape=(self.max_expected_steps, n_e, self.n_q, 6), chunks=(1, n_e, self.n_q, 6), dtype='f4', overwrite=True)
        self.z_U = self.z_root.create_array('U', shape=(self.max_expected_steps, n_n, 3), chunks=(1, n_n, 3), dtype='f4', overwrite=True)
        self.z_temp = self.z_root.create_array('temperature', shape=(self.max_expected_steps, n_n), chunks=(1, n_n), dtype='f4', overwrite=True)
        self.z_time = self.z_root.create_array('time', shape=(self.max_expected_steps,), chunks=(100,), dtype='f4', overwrite=True)
        
        self.save_step = 0

    def _solve_mechanical_step(self, n_e_active, n_n_active, temp_ip, B, D_elast, iD, jD, ele_detJac, free, mask, K_diag):
        """
        Executes the non-linear Newton-Raphson loop using Matrix-Free Conjugate Gradient.
        """
        # Interpolate Properties exactly as the original code did (no stiffness clipping)
        young = cp.interp(temp_ip, self.temp_young1, self.young1)
        shear = young / (2 * (1 + self.poisson))
        bulk = young / (3 * (1 - 2 * self.poisson))
        Y = cp.interp(temp_ip, self.temp_Y1, self.Y1)
        scl = cp.interp(temp_ip, self.temp_scl1, self.scl1)
        a = self.a1 * cp.ones_like(young)
        
        self.alpha_Th[:, :, 0:3] = scl[:, :, cp.newaxis].repeat(3, axis=2)
        
        Ep_converged, Hard_converged = None, None
        v_full = cp.zeros(3 * n_n_active, dtype=cp.float64)
        dUv = cp.zeros(3 * n_n_active, dtype=cp.float64)
        
        # 2. Setup Jacobi Preconditioner
        M_inv = 1.0 / (K_diag[free] + 1e-12) 
        
        def precond_matvec(v):
            return M_inv * v
            
        M_op = LinearOperator((len(free), len(free)), matvec=precond_matvec)

        for beta in [1.0, 0.5, 0.3, 0.1]:
            U_it = self.U[0:n_n_active]
            
            for it in range(self.Maxit):
                # 1. Strain Predictor
                self.E[0:n_e_active] = cp.reshape(B @ U_it.flatten(), (-1, 8, 6))
                self.E[0:n_e_active] -= (temp_ip[0:n_e_active, :, cp.newaxis].repeat(6, axis=2) - self.domain.ambient) * self.alpha_Th[0:n_e_active]
                
                # 2. Constitutive Problem (Radial Return)
                S_iter, DS, IND_p, Ep_iter, Hard_iter = constitutive_problem(
                    self.E[0:n_e_active], self.Ep_prev[0:n_e_active], self.Hard_prev[0:n_e_active],
                    shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active]
                )
                
                # 3. Tangent Mapping
                vD = ele_detJac[:, :, cp.newaxis, cp.newaxis].repeat(6, axis=2).repeat(6, axis=3) * DS
                D_p = cusparse.csr_matrix((vD.ravel(), (iD.ravel(), jD.ravel())), shape=D_elast.shape)
                
                F_dof = B.transpose() @ ((ele_detJac[:, :, cp.newaxis].repeat(6, axis=2) * S_iter).reshape(-1))
                b = -F_dof[free]

                # 4. Matrix-Free CG Solver Setup
                it_count = 0
                def monitor(xk):
                    nonlocal it_count
                    it_count += 1
                    if it_count % 500 == 0:
                        current_res = cp.linalg.norm(b - A_op.matvec(xk)) / (cp.linalg.norm(b) + 1e-12)
                        print(f"      [CG Iter {it_count}] Rel. Residual: {float(current_res):.2e}", flush=True)

                def matvec_free(v_free):
                    v_full.fill(0)
                    v_full[free] = v_free
                    temp1 = B.dot(v_full)
                    temp2 = D_p.dot(temp1)
                    return (B.transpose().dot(temp2))[free]

                A_op = LinearOperator((len(free), len(free)), matvec=matvec_free)
                
                # I kept the maxiter safeguard here to prevent infinite silent hangs if your material text file
                # evaluates to a strictly singular matrix at 2300K. This does not alter the physics.
                x, info = cusparse_linalg.cg(A_op, b, M=M_op, rtol=self.tol, maxiter=10000, callback=monitor)
                
                if info > 0:
                    print(f"    !!! CG hit maxiter (10000) without reaching tolerance.", flush=True)
                elif info < 0:
                    print(f"    !!! CG failed with mathematical breakdown.", flush=True)

                # 5. Corrector
                dUv = cp.zeros(3 * n_n_active, dtype=F_dof.dtype)
                dUv[free] = x
                U_new = U_it + beta * dUv.reshape(n_n_active, 3)

                # 6. Matrix-Free Convergence Check
                def compute_energy(vec):
                    strain = B.dot(vec)
                    stress = D_elast.dot(strain)
                    energy = strain.dot(stress)
                    del strain, stress
                    return energy

                q1 = beta**2 * compute_energy(dUv)
                q2 = compute_energy(U_it.flatten())
                q3 = compute_energy(U_new.flatten())
                
                del D_p, A_op, dUv, b, x, vD, F_dof
                
                if (q1 / (q2 + q3 + 1e-12)) < self.tol:
                    U_it = U_new
                    Ep_converged, Hard_converged = Ep_iter, Hard_iter
                    break
                U_it = U_new
            else:
                continue
            break
        else:
            raise Exception(f'Newton solver failed at t={self.domain.current_sim_time}')
            
        return U_it, S_iter, Ep_converged, Hard_converged

    def run(self, params, generator, active_print_time):
        """Executes the coupled time loop."""
        print(f"Step 1: Loading domain and toolpath...", flush=True)
        self.domain = domain_mgr(filename=self.geom_file, input_data_dir=self.input_dir, verbose=False)
        self.heat_solver = heat_solve_mgr(self.domain)
        
        print(f"Step 2: Preparing material arrays...", flush=True)
        self._load_materials()
        self._init_state_arrays()
        self.cumulative_ht = cp.zeros(self.domain.nN, dtype=cp.float64)
        T_MIN_HT = 654.0 + 273.15 
        T_MAX_HT = 857.0 + 273.15
        T_LIQUIDUS = 1336.0 + 273.15
        generator.generate_profile(params)
        
        # Precompute boundary mask
        min_z = float(cp.min(self.domain.nodes[:, 2]))
        idirich = cp.array(cp.abs(self.domain.nodes[:, 2] - min_z) < 1e-5)
        
        n_e_old = int(cp.sum(self.domain.element_birth < 1e-5))
        n_n_old = int(cp.sum(self.domain.node_birth < 1e-5))
        last_mech_time = 0
        K_diag, B, D_elast, iD, jD, ele_detJac = None, None, None, None, None, None
        
        total_sim_time = self.domain.end_sim_time
        cooling_duration = total_sim_time - active_print_time
                
        print(f"Starting Simulation Loop (Printing: {active_print_time}s | Cooling: {cooling_duration}s | Total: {self.domain.end_sim_time}s)...", flush=True)
        
        while self.domain.current_sim_time < self.domain.end_sim_time - self.domain.dt:
            prev_T = self.heat_solver.temperature.copy()

            # --- THERMAL INTEGRATION ---
            if self.domain.current_sim_time <= active_print_time:
                current_power = generator.get_power_at_time(self.domain.current_sim_time)
            else:
                current_power = 0.0 
                
            self.heat_solver.q_in = current_power * self.domain.absortivity
            self.heat_solver.time_integration()
            current_T = self.heat_solver.temperature
            is_liquidus = current_T >= T_LIQUIDUS
            self.cumulative_ht[is_liquidus] = 0.0

            # --- CONTINUOUS FRACTIONAL HEAT TREATMENT ---
            # 1. Fully Inside the Band
            fully_in = (prev_T >= T_MIN_HT) & (prev_T <= T_MAX_HT) & (current_T >= T_MIN_HT) & (current_T <= T_MAX_HT)
            self.cumulative_ht[fully_in] += self.domain.dt
            
            # 2. Crossing the Lower Boundary (654 C)
            cross_up_min = (prev_T < T_MIN_HT) & (current_T >= T_MIN_HT)
            if cp.any(cross_up_min):
                fraction_in = (current_T[cross_up_min] - T_MIN_HT) / (current_T[cross_up_min] - prev_T[cross_up_min])
                self.cumulative_ht[cross_up_min] += fraction_in * self.domain.dt
                
            cross_down_min = (prev_T >= T_MIN_HT) & (current_T < T_MIN_HT)
            if cp.any(cross_down_min):
                fraction_in = (prev_T[cross_down_min] - T_MIN_HT) / (prev_T[cross_down_min] - current_T[cross_down_min])
                self.cumulative_ht[cross_down_min] += fraction_in * self.domain.dt

            # 3. Crossing the Upper Boundary (857 C)
            cross_up_max = (prev_T <= T_MAX_HT) & (current_T > T_MAX_HT)
            if cp.any(cross_up_max):
                fraction_in = (T_MAX_HT - prev_T[cross_up_max]) / (current_T[cross_up_max] - prev_T[cross_up_max])
                self.cumulative_ht[cross_up_max] += fraction_in * self.domain.dt
                
            cross_down_max = (prev_T > T_MAX_HT) & (current_T <= T_MAX_HT)
            if cp.any(cross_down_max):
                fraction_in = (T_MAX_HT - current_T[cross_down_max]) / (prev_T[cross_down_max] - current_T[cross_down_max])
                self.cumulative_ht[cross_down_max] += fraction_in * self.domain.dt

            # --- THERMAL HEARTBEAT ---
            if self.heat_solver.current_step % 2000 == 0:
                progress = (self.domain.current_sim_time / self.domain.end_sim_time) * 100
                print(f"  [Thermal] Time: {self.domain.current_sim_time:.3f}s / {self.domain.end_sim_time}s ({progress:.1f}%)", flush=True)

            n_e_active = int(cp.sum(self.domain.element_birth < self.domain.current_sim_time))
            n_n_active = int(cp.sum(self.domain.node_birth < self.domain.current_sim_time))
            
            if self.domain.current_sim_time > active_print_time:
                implicit_timestep = 0.5  # Take massive 0.5s strides during cooling
            else:
                implicit_timestep = 0.02 # Keep high resolution 0.02s strides during printing
            
            # Mechanical Step
            if self.domain.current_sim_time >= last_mech_time + implicit_timestep:
                # --- MECHANICAL HEARTBEAT ---
                print(f"  >>> MECHANICAL SOLVE: Time {self.domain.current_sim_time:.3f}s | Active Eles: {n_e_active}", flush=True)

                # Cast element/node arrays to GPU before applying masks
                elements_gpu = cp.asarray(self.domain.elements)
                nodes_gpu = cp.asarray(self.domain.nodes)
                
                active_eles = elements_gpu[0:n_e_active]
                active_nodes = nodes_gpu[0:n_n_active]
                
                if n_n_active > n_n_old:
                    self.U = disp_match(self.domain.nodes, self.U, n_n_old, self.domain.nN)
                    
                temp_ip = (self.domain.Nip_ele[:,cp.newaxis,:] @ self.heat_solver.temperature[elements_gpu][:,cp.newaxis,:,cp.newaxis].repeat(8,axis=1))[:,:,0,0]
                
                # --- ORIGINAL CLIPPING ---
                # Only the temperature is clipped to 2300K, exactly as in the original code.
                temp_ip = cp.clip(temp_ip, 300, 2300)
                
                if n_e_active > n_e_old or K_diag is None:
                    del K_diag, B, D_elast
                    # cp.get_default_memory_pool().free_all_blocks()
                    
                    young_base = cp.interp(temp_ip, self.temp_young1, self.young1)
                    shear_base = young_base / (2*(1+self.poisson))
                    bulk_base = young_base / (3*(1-2*self.poisson))
                    
                    K_diag, B, D_elast, _, _, iD, jD, ele_detJac = elastic_stiff_matrix(
                        active_eles, active_nodes, self.domain.Bip_ele, shear_base[0:n_e_active], bulk_base[0:n_e_active]
                    )

                # Apply Boundary Conditions
                Q_mask = cp.ones((n_n_active, 3), dtype=bool)
                Q_mask[idirich[0:n_n_active], :] = False
                mask = Q_mask.flatten()
                free = cp.where(mask)[0]

                # Run Non-Linear Solver
                U_it, S_iter, Ep_conv, Hard_conv = self._solve_mechanical_step(
                    n_e_active, n_n_active, temp_ip, B, D_elast, iD, jD, ele_detJac, free, mask, K_diag
                )
                # Update Global States
                self.U[0:n_n_active], self.S[0:n_e_active] = U_it, S_iter
                self.Ep_prev[0:n_e_active], self.Hard_prev[0:n_e_active] = Ep_conv, Hard_conv
                
                # --- OPTIMIZED ZARR WRITE ---
                # --- OPTIMIZED ZARR WRITE ---
                if self.save_step < self.max_expected_steps:
                    # Cast to float32 ON THE GPU before crossing the PCIe bus
                    cpu_S_active = self.S[:n_e_active].astype(cp.float32).get()
                    cpu_U_active = self.U[:n_n_active].astype(cp.float32).get()
                    
                    self.z_stress[self.save_step, :n_e_active, :, :] = cpu_S_active
                    self.z_U[self.save_step, :n_n_active, :] = cpu_U_active
                    self.z_temp[self.save_step, :] = self.heat_solver.temperature.astype(cp.float32).get()
                    self.z_time[self.save_step] = self.domain.current_sim_time
                    
                    self.save_step += 1
                else:
                    print("WARNING: Zarr array max steps exceeded. Skipping save.", flush=True)

                n_e_old, n_n_old, last_mech_time = n_e_active, n_n_active, self.domain.current_sim_time

        # --- TRUNCATE ZARR ARRAYS ---
        # Strip away unused allocated time steps so Paraview doesn't read empty frames
        self.z_stress.resize((self.save_step, self.domain.nE, self.n_q, 6))
        self.z_U.resize((self.save_step, self.domain.nN, 3))
        self.z_temp.resize((self.save_step, self.domain.nN))
        self.z_time.resize((self.save_step,))

        # ==========================================
        # OBJECTIVE EVALUATION: DYNAMIC CORE EXTRACTION
        # ==========================================
        node_birth_gpu = cp.asarray(self.domain.node_birth)
        elem_birth_gpu = cp.asarray(self.domain.element_birth)
        # 1. Isolate ONLY the printed metal 
        # Must be born AFTER t=0 (no substrate) AND BEFORE the laser turned off (no ghost layers)
        is_node_born = (node_birth_gpu > 1e-5) & (node_birth_gpu <= active_print_time)
        is_elem_born = (elem_birth_gpu > 1e-5) & (elem_birth_gpu <= active_print_time)
        
        z_coords = cp.asarray(self.domain.nodes[:, 2])
        
        # 2. Find unique Z-planes of ONLY the valid printed metal
        active_z_coords = z_coords[is_node_born]
        unique_z = cp.unique(cp.around(active_z_coords, decimals=4))
        
        # 3. EXCLUDE: Top TWO layers (z[-1], z[-2]) of the printed part
        if len(unique_z) > 2:
            z_min_bound = unique_z[0]  # The absolute bottom layer of the printed metal
            z_max_bound = unique_z[-3] # Highest layer beneath the true top two
            
            # The final valid mask: Must be printed metal AND inside the safe Z-bounds
            active_node_mask = is_node_born & (z_coords >= z_min_bound - 1e-4) & (z_coords <= z_max_bound + 1e-4)
            
            # Element heights are the average of their nodes
            elem_z = cp.mean(z_coords[cp.asarray(self.domain.elements)], axis=1)
            active_elem_mask = is_elem_born & (elem_z >= z_min_bound - 1e-4) & (elem_z <= z_max_bound + 1e-4)
        else:
            # Fallback if the print was so short it only has 1 or 2 layers
            active_node_mask = is_node_born
            active_elem_mask = is_elem_born

        # ==========================================
        # 4. CALCULATE METRICS ON THE ISOLATED CORE
        # ==========================================
        # RESIDUAL STRESS
        S_active = self.S[active_elem_mask]
        if len(S_active) > 0:
            S_vm = cp.sqrt(0.5 * ((S_active[:,:,0]-S_active[:,:,1])**2 + (S_active[:,:,1]-S_active[:,:,2])**2 + 
                                  (S_active[:,:,2]-S_active[:,:,0])**2 + 6*(S_active[:,:,3]**2 + S_active[:,:,4]**2 + S_active[:,:,5]**2)))
            max_residual_stress = float(cp.max(S_vm)) 
            avg_residual_stress = float(cp.mean(S_vm))
        else:
            max_residual_stress, avg_residual_stress = 9999.0, 9999.0

        # HEAT TREATMENT TIME
        active_ht_durations = self.cumulative_ht[active_node_mask]
        if len(active_ht_durations) > 0:
            avg_heat_treatment_time = float(cp.mean(active_ht_durations))
            min_heat_treatment_time = float(cp.min(active_ht_durations)) 
        else:
            avg_heat_treatment_time, min_heat_treatment_time = 0.0, 0.0

        # MAXIMUM DISTORTION (WARPAGE)
        U_active = self.U[active_node_mask]
        if len(U_active) > 0:
            displacement_magnitudes = cp.linalg.norm(U_active, axis=1)
            max_warpage = float(cp.max(displacement_magnitudes))
        else:
            max_warpage = 9999.0

        # EQUIVALENT PLASTIC STRAIN (PEEQ)
        Ep_active = self.Ep_prev[active_elem_mask]
        if len(Ep_active) > 0:
            peeq = cp.sqrt((2.0/3.0) * (Ep_active[:,:,0]**2 + Ep_active[:,:,1]**2 + Ep_active[:,:,2]**2 + 
                                       0.5 * (Ep_active[:,:,3]**2 + Ep_active[:,:,4]**2 + Ep_active[:,:,5]**2)))
            max_peeq = float(cp.max(peeq))
        else:
            max_peeq = 9999.0

        return max_residual_stress, avg_residual_stress, avg_heat_treatment_time, min_heat_treatment_time, max_warpage, max_peeq


def run_coupled_simulation(params, generator, input_dir="../0_properties", geom_file='wall.k', output_path="stress_history.zarr", active_print_time=40.0):
    """
    Wrapper function to maintain compatibility with existing BO loops.
    """
    try:
        sim = DEDSimulator(input_dir=input_dir, geom_file=geom_file, output_path=output_path)
        objective = sim.run(params, generator, active_print_time=active_print_time)
        return objective
    finally:
        # Guarantee massive GPU arrays are wiped out even if the simulation crashes
        print("Cleaning up GPU memory...", flush=True)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()