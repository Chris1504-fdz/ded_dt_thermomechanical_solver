import os
import torch
import numpy as np
import concurrent.futures
from scipy.stats import qmc

# BoTorch Imports
from botorch.models import ModelListGP, SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf

# --- CONFIGURATION ---
SIM_DURATION = 34.1
PROPERTIES_DIR = "."
GEOM_FILE = "thinwall.k"
NUM_PARAMS = 10

GPU_MAPPING = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
BATCH_SIZE = len(GPU_MAPPING)

def isolated_simulation_worker(task_index, params_np):
    """
    Strictly sandboxed worker. Blinds the process to other GPUs before CuPy loads.
    """
    gpu_id = GPU_MAPPING[task_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # CuPy and custom modules MUST be imported inside this function
    import cupy as cp
    from laser_generator import LaserProfileGenerator
    from thermomechanical import run_coupled_simulation
    
    # Force CuPy to use the only device it can see (which is internally ID 0)
    cp.cuda.Device(0).use()
    
    print(f"[Worker {task_index}] Initializing on physical GPU {gpu_id}...")
    
    # 1. Initialize Laser Generator
    generator = LaserProfileGenerator(total_time=SIM_DURATION)
    generator.generate_profile(params_np)
    
    # 2. Unique Zarr output to prevent parallel overwrite collisions
    unique_zarr = f"stress_history_task_{task_index}.zarr"
    
    # 3. Run the physics
    try:
        max_stress, avg_stress, avg_ht, min_ht = run_coupled_simulation(
            params=params_np,
            generator=generator,
            input_dir=PROPERTIES_DIR,
            geom_file=GEOM_FILE,
            output_path=unique_zarr,
            active_print_time=SIM_DURATION
        )
    except Exception as e:
        print(f"[!] Worker {task_index} FAILED: {e}")
        # Return a heavily penalized result if the simulation crashes
        max_stress, avg_stress, avg_ht, min_ht = 2000.0, 1500.0, 0.0, 0.0
    
    # 4. Flush GPU memory before the process returns
    cp.get_default_memory_pool().free_all_blocks()
    
    # Return formatted for BoTorch: [Obj 1 (Maximize), Obj 2 (Maximize), Constraint 1, Constraint 2]
    return [-avg_stress, avg_ht, min_ht, max_stress]

def generate_initial_data(num_samples):
    """Generates initial dataset using Latin Hypercube Sampling across GPUs."""
    print(f"\n--- GENERATING INITIAL DATASET ({num_samples} SAMPLES) ---")
    sampler = qmc.LatinHypercube(d=NUM_PARAMS)
    init_X_np = sampler.random(n=num_samples)
    
    results = [None] * num_samples
    
    # Map the initial samples to the GPUs using the same isolated worker
    with concurrent.futures.ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_idx = {
            executor.submit(isolated_simulation_worker, i % BATCH_SIZE, init_X_np[i]): i 
            for i in range(num_samples)
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            print(f"  [+] Initial Sample {idx+1}/{num_samples} completed.")

    return torch.tensor(init_X_np, dtype=torch.float64), torch.tensor(results, dtype=torch.float64)

def run_optimization(train_X, train_Y, num_iterations=10):
    bounds = torch.tensor([[0.0] * NUM_PARAMS, [1.0] * NUM_PARAMS], dtype=torch.float64)
    
    T_MIN = 15.0     
    S_MAX = 1200.0   
    constraints = [
        lambda Z: T_MIN - Z[..., 2], 
        lambda Z: Z[..., 3] - S_MAX   
    ]
    
    ref_point = torch.tensor([-1500.0, 0.0], dtype=torch.float64)

    for iteration in range(num_iterations):
        print(f"\n{'='*40}")
        print(f" BO ITERATION {iteration + 1} / {num_iterations}")
        print(f"{'='*40}")
        
        print("Training GP Models...")
        models = [SingleTaskGP(train_X, train_Y[..., i:i+1]) for i in range(4)]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        obj_Y = train_Y[..., 0:2]
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=obj_Y)
        
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            constraints=constraints,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1])
        )
        
        print(f"Solving Acquisition Function for {BATCH_SIZE} candidates...")
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=BATCH_SIZE, 
            num_restarts=10,
            raw_samples=512,
            sequential=True
        )
        
        print(f"Dispatching physics simulations to GPUs...")
        new_results = [None] * BATCH_SIZE
        candidates_np = candidates.detach().numpy()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_idx = {
                executor.submit(isolated_simulation_worker, i, candidates_np[i]): i 
                for i in range(BATCH_SIZE)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                new_results[idx] = future.result()
                print(f"  [+] BO Task {idx:02d} (GPU {GPU_MAPPING[idx]}) finished.")
        
        new_Y = torch.tensor(new_results, dtype=torch.float64)
        train_X = torch.cat([train_X, candidates])
        train_Y = torch.cat([train_Y, new_Y])
        
        torch.save({'train_X': train_X, 'train_Y': train_Y}, f"bo_checkpoint_iter_{iteration+1}.pt")
        print(f"Iteration complete. Data saved.")

    return train_X, train_Y

if __name__ == '__main__':
    # Generate 11 initial random samples using Latin Hypercube
    init_X, init_Y = generate_initial_data(num_samples=BATCH_SIZE)
    
    # Run the optimization loop
    run_optimization(init_X, init_Y, num_iterations=20)