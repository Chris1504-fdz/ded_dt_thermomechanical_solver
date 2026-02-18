import cupy as cp
import numpy as np
import zarr
import os
import time
import gc
import cupyx.scipy.sparse as cusparse
from cupyx.scipy.sparse import linalg as cusparse_linalg
from cupyx.scipy.sparse.linalg import LinearOperator
# Import Gamma Core Modules
from gamma.simulator.gamma import domain_mgr, heat_solve_mgr
# Import Mechanical Modules
from gamma.simulator.func import elastic_stiff_matrix, constitutive_problem, disp_match
from laser_generator import LaserProfileGenerator

def run_coupled_simulation(params, generator, input_dir="../0_properties", geom_file='wall.k', output_path="stress_history.zarr"):
    """
    Runs a fully coupled thermo-mechanical simulation with memory-optimized assembly.
    Uses float64 (Double Precision) for maximum algorithmic correctness.
    """
    abs_input_dir = os.path.abspath(input_dir)
    
    # --- MEMORY DIAGNOSTICS ---
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    free_mem, total_mem = cp.cuda.Device().mem_info
    print(f"DEBUG: Initial GPU Memory: Free={(free_mem/1024**3):.2f}GB / Total={(total_mem/1024**3):.2f}GB")
    
    if os.path.exists(abs_input_dir):
        files = os.listdir(abs_input_dir)
        print(f"DEBUG: Found {len(files)} files in properties folder.")
    else:
        print(f"ERROR: Directory {abs_input_dir} does not exist!")
        return None
    
    # 1. SETUP DOMAIN & SOLVERS
    domain = domain_mgr(filename=geom_file, input_data_dir=abs_input_dir, verbose=True) 
    heat_solver = heat_solve_mgr(domain)
    
    # Mechanical Constants
    n_n = len(domain.nodes)
    n_e = len(domain.elements)
    n_q = 8 # Integration points
    
    print(f"\n--- MESH STATISTICS FOR {geom_file} ---")
    print(f"Nodes: {n_n}")
    print(f"Elements: {n_e}")
    est_matrix_mem = (n_e * 24 * 24 * 8 * 8) / (1024**3)
    print(f"Estimated Stiffness Matrix Block Size: ~{est_matrix_mem:.2f} GB (Uncompressed float64)")
    print("---------------------------------------\n")
    
    # --- ZARR INITIALIZATION ---
    z_root = zarr.open(output_path, mode='w')
    
    # Create datasets for Stress, Displacement (U), Temperature, and Time
    # Stored as float32 ('f4') to minimize disk bloat
    z_stress = z_root.create_array('stress', shape=(0, n_e, n_q, 6), chunks=(1, n_e, n_q, 6), dtype='f4', overwrite=True)
    z_U = z_root.create_array('U', shape=(0, n_n, 3), chunks=(1, n_n, 3), dtype='f4', overwrite=True)
    z_temp = z_root.create_array('temperature', shape=(0, n_n), chunks=(1, n_n), dtype='f4', overwrite=True)
    z_time = z_root.create_array('time', shape=(0,), chunks=(100,), dtype='f4', overwrite=True)

    try:
        # Material Props
        young1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Young_Debroy.txt'))[:,1]/1e6)
        temp_young1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Young_Debroy.txt'))[:,0])
        Y1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Yield_Debroy.txt'))[:,1]/1e6*np.sqrt(2/3))
        temp_Y1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Yield_Debroy.txt'))[:,0])
        scl1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Alpha_Debroy.txt'))[:,1])
        temp_scl1 = cp.array(np.loadtxt(os.path.join(abs_input_dir, 'materials/TI64_Alpha_Debroy.txt'))[:,0])
        
        poisson = 0.3
        a1 = 10000
        T_Ref = domain.ambient

        # Initialization of Mechanical Arrays
        E = cp.zeros((n_e, n_q, 6))       
        S = cp.zeros((n_e, n_q, 6))       
        Ep_prev = cp.zeros((n_e, n_q, 6)) 
        Hard_prev = cp.zeros((n_e, n_q, 6))
        U = cp.zeros((n_n, 3))            
        alpha_Th = cp.zeros((n_e, n_q, 6))
        
        # DYNAMIC BOUNDARY CONDITION
        min_z = float(cp.min(domain.nodes[:, 2]))
        idirich = cp.array(cp.abs(domain.nodes[:, 2] - min_z) < 1e-5)


        n_e_old = int(cp.sum(domain.element_birth < 1e-5))
        n_n_old = int(cp.sum(domain.node_birth < 1e-5))
        
        tol = 1.0e-8 
        Maxit = 20

        # Cache for Stiffness Matrix
        K_elast, B, D_elast, iD, jD, ele_detJac_active = None, None, None, None, None, None

        # --- LASER GENERATOR PRECOMPUTATION ---
        generator.generate_profile(params)

        # 2. TIME LOOP
        last_mech_time = 0
        mech_step_count = 0
        
        print(f"Starting Simulation Loop (Total Time: {domain.end_sim_time}s)...")
        
        while domain.current_sim_time < domain.end_sim_time - domain.dt:
            current_power = generator.get_power_at_time(domain.current_sim_time)
            heat_solver.q_in = current_power * domain.absortivity
            heat_solver.time_integration()
            
            n_e_active = int(cp.sum(domain.element_birth < domain.current_sim_time))
            n_n_active = int(cp.sum(domain.node_birth < domain.current_sim_time)) 
            
            implicit_timestep = 0.1 if (heat_solver.laser_state == 0 and n_e_active == n_e_old) else 0.02
                
            if domain.current_sim_time >= last_mech_time + implicit_timestep:
                mech_step_count += 1
                
                if mech_step_count % 10 == 0:
                    print(f"  > Sim Time: {domain.current_sim_time:.2f}s | Active Elements: {n_e_active}")

                mempool.free_all_blocks()

                active_eles = domain.elements[0:n_e_active]
                active_nodes = domain.nodes[0:n_n_active]
                
                if n_n_active > n_n_old:
                    U = disp_match(domain.nodes, U, n_n_old, n_n)
                
                temperature_ip = (domain.Nip_ele[:,cp.newaxis,:] @ heat_solver.temperature[domain.elements][:,cp.newaxis,:,cp.newaxis].repeat(8,axis=1))[:,:,0,0]
                temperature_ip = cp.clip(temperature_ip, 300, 2300)
                
                young = cp.interp(temperature_ip, temp_young1, young1)
                shear = young/(2*(1+poisson))
                bulk = young/(3*(1-2*poisson))
                scl = cp.interp(temperature_ip, temp_scl1, scl1)
                alpha_Th[:,:,0:3] = scl[:,:,cp.newaxis].repeat(3, axis=2)
                Y = cp.interp(temperature_ip, temp_Y1, Y1)
                a = a1 * cp.ones_like(young)
                
                if n_e_active > n_e_old or K_elast is None:
                    del K_elast, B, D_elast
                    mempool.free_all_blocks()
                    
                    K_elast, B, D_elast, _, _, iD, jD, ele_detJac_active = elastic_stiff_matrix(
                        active_eles, active_nodes, domain.Bip_ele, shear[0:n_e_active], bulk[0:n_e_active]
                    )

                Q = cp.zeros((n_n_active, 3), dtype=bool)
                Q[:, :] = True
                Q[idirich[0:n_n_active], :] = False
                mask = Q.flatten()
                free = cp.where(mask)[0]

                Ep_converged, Hard_converged = None, None

                for beta in [1.0, 0.5, 0.3, 0.1]:
                    U_it = U[0:n_n_active]
                    for it in range(Maxit):
                        E[0:n_e_active] = cp.reshape(B @ U_it.flatten(), (-1, 8, 6))
                        E[0:n_e_active] -= (temperature_ip[0:n_e_active,:,cp.newaxis].repeat(6,axis=2) - T_Ref) * alpha_Th[0:n_e_active]
                        
                        S_iter, DS, IND_p, Ep_iter, Hard_iter = constitutive_problem(
                            E[0:n_e_active], Ep_prev[0:n_e_active], Hard_prev[0:n_e_active],
                            shear[0:n_e_active], bulk[0:n_e_active], a[0:n_e_active], Y[0:n_e_active]
                        )
                        
                        vD = ele_detJac_active[:,:,cp.newaxis,cp.newaxis].repeat(6,axis=2).repeat(6,axis=3) * DS
                        D_p = cusparse.csr_matrix((cp.ndarray.flatten(vD), (cp.ndarray.flatten(iD), cp.ndarray.flatten(jD))), shape=D_elast.shape)
                        
                        F_dof = B.transpose() @ ((ele_detJac_active[:, :, cp.newaxis].repeat(6, axis=2) * S_iter).reshape(-1))
                        b = -F_dof[free]

                        # --- THE MATRIX-FREE TRICK ---
                        D_delta = D_p - D_elast
                        
                        # Define how K_tangent * vector behaves without forming K_tangent
                        def matvec_free(v_free):
                            # 1. Map the free DOFs back to a full-size vector
                            v_full = cp.zeros(3 * n_n_active, dtype=v_free.dtype)
                            v_full[free] = v_free
                            
                            # 2. Sequential Matrix-Vector multiplications (Extremely memory efficient)
                            temp1 = B.dot(v_full)
                            temp2 = D_delta.dot(temp1)
                            temp3 = B.transpose().dot(temp2)
                            
                            # 3. Add the elastic stiffness contribution
                            res_full = K_elast.dot(v_full) + temp3
                            
                            # 4. Return only the free DOFs
                            return res_full[free]
                        
                        # Wrap the function in a LinearOperator
                        n_free = len(free)
                        A_op = LinearOperator((n_free, n_free), matvec=matvec_free)

                        # Pass the LinearOperator directly into the CG solver
                        x, info = cusparse_linalg.cg(A_op, b, rtol=tol)
                        
                        del D_delta, A_op
                        # -----------------------------

                        dUv = cp.zeros(3 * n_n_active, dtype=F_dof.dtype)
                        dUv[free] = x
                        U_new = U_it + beta * dUv.reshape(n_n_active, 3)

                        q1 = beta**2 * dUv.dot(K_elast.dot(dUv))
                        q2 = U_it.flatten().dot(K_elast.dot(U_it.flatten()))
                        q3 = U_new.flatten().dot(K_elast.dot(U_new.flatten()))
                        
                        del D_p, dUv, b, x
                        
                        if (q1 / (q2 + q3 + 1e-12)) < tol:
                            U_it = U_new
                            Ep_converged = Ep_iter
                            Hard_converged = Hard_iter
                            break
                        U_it = U_new
                    else: continue
                    break
                else: raise Exception(f'Newton solver failed at t={domain.current_sim_time}')

                U[0:n_n_active], S[0:n_e_active] = U_it, S_iter
                Ep_prev[0:n_e_active], Hard_prev[0:n_e_active] = Ep_converged, Hard_converged
                
                # --- PERSIST TO ZARR ---
                # Move to CPU and immediately append global states
                z_stress.append(S.get()[np.newaxis, ...], axis=0)
                z_U.append(U.get()[np.newaxis, ...], axis=0)
                z_temp.append(heat_solver.temperature.get()[np.newaxis, ...], axis=0)
                z_time.append(np.array([domain.current_sim_time], dtype='f4'))

                n_e_old, n_n_old, last_mech_time = n_e_active, n_n_active, domain.current_sim_time

        # 3. CALCULATE FINAL OBJECTIVE
        S_final = S[0:n_e_active]
        S_vm = cp.sqrt(0.5 * ((S_final[:,:,0]-S_final[:,:,1])**2 + (S_final[:,:,1]-S_final[:,:,2])**2 + 
                              (S_final[:,:,2]-S_final[:,:,0])**2 + 6*(S_final[:,:,3]**2 + S_final[:,:,4]**2 + S_final[:,:,5]**2)))
        return float(cp.max(S_vm))

    finally:
        print("Cleaning up GPU memory...")
        vars_to_delete = ['E', 'S', 'Ep_prev', 'Hard_prev', 'U', 'alpha_Th', 
                          'K_elast', 'B', 'D_elast', 'domain', 'heat_solver', 'K_tangent', 'D_p']
        for var in vars_to_delete:
            if var in locals():
                del locals()[var]
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("GPU memory cleared.")