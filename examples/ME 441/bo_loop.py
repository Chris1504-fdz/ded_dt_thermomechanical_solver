import torch
import numpy as np
import os
import multiprocessing
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
import cupy as cp

# Import your custom modules
from laser_generator import LaserProfileGenerator
from thermomechanical import run_coupled_simulation

# --- CONFIGURATION ---
N_INITIAL = 8           # Initial random simulations (will run in 2 batches of 4)
N_ITERATIONS = 20       # Number of BO rounds
N_BATCH = 4             # Parallel simulations (Matching 4 GPUs)
N_PARAMS = 10           # Your fourier parameters
SIM_DURATION = 12.854   # The actual time in your .crs file

# GPU List to utilize
GPULIST = [0, 1, 2, 3]

# Directories - ADJUST THESE TO YOUR ACTUAL ENVIRONMENT
PROPERTIES_DIR = "../0_properties"
GEOM_FILE = "thinwall.k"

def simulation_worker(gpu_id, params, sim_duration, properties_dir, geom_file):
    import gc # Import inside worker
    try:
        with cp.cuda.Device(gpu_id):
            generator = LaserProfileGenerator(total_time=sim_duration)
            
            # Use a deterministic label for debugging
            iteration_label = f"gpu{gpu_id}_{time.time()}"
            zarr_path = f"stress_history_{iteration_label}.zarr"
            
            val = run_coupled_simulation(
                params=params,
                generator=generator,
                input_dir=properties_dir,
                geom_file=geom_file,
                output_path=zarr_path
            )
            
            # Cleanup GPU memory explicitly before exiting process
            cp.get_default_memory_pool().free_all_blocks()
            return -float(val)
    except Exception as e:
        print(f"CRITICAL: GPU {gpu_id} failed: {e}")
        return None # Handle as NaN in the main loop

def evaluate_batch_parallel(candidates_tensor):
    """
    Dispatches a batch of parameters to the available GPUs in parallel.
    """
    params_list = candidates_tensor.numpy()
    args = []
    
    for i in range(len(params_list)):
        # Assign GPU based on index in batch
        gpu_id = GPULIST[i % len(GPULIST)]
        args.append((gpu_id, params_list[i], SIM_DURATION, PROPERTIES_DIR, GEOM_FILE))
    
    # Use multiprocessing Pool to execute on 4 GPUs simultaneously
    with multiprocessing.Pool(processes=len(params_list)) as pool:
        results = pool.starmap(simulation_worker, args)
        
    return torch.tensor(results).double().unsqueeze(-1)

# --- BO EXECUTION ---

if __name__ == "__main__":
    # Ensure multiprocessing works correctly with CUDA
    multiprocessing.set_start_method('spawn', force=True)

    # 1. Initialization: Generate Initial Random Data in batches of 4
    print(f"Step 1: Running {N_INITIAL} initial random samples in parallel batches...")
    train_X = torch.rand(N_INITIAL, N_PARAMS).double()
    train_Y_list = []
    
    for i in range(0, N_INITIAL, N_BATCH):
        batch_X = train_X[i:i+N_BATCH]
        print(f"  Evaluating initial batch {i//N_BATCH + 1}...")
        batch_Y = evaluate_batch_parallel(batch_X)
        train_Y_list.append(batch_Y)
        
    train_Y = torch.cat(train_Y_list)

    # 2. Main Optimization Loop
    for iteration in range(N_ITERATIONS):
        print(f"\n--- BO Iteration {iteration + 1}/{N_ITERATIONS} ---")
        
        # A. Fit the Surrogate Model (Gaussian Process)
        gp = SingleTaskGP(train_X, standardize(train_Y))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # B. Propose Next Best Parameters (Acquisition Function)
        # qExpectedImprovement handles the batch of 4 candidates
        acq_func = qExpectedImprovement(
            model=gp,
            best_f=train_Y.max(),
        )
        
        new_candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(N_PARAMS), torch.ones(N_PARAMS)]),
            q=N_BATCH,
            num_restarts=10,
            raw_samples=512,
        )
        
        # C. Evaluate the Candidates in Parallel across 4 GPUs
        print(f"  Dispatching batch of {N_BATCH} to GPUs {GPULIST}...")
        new_Y = evaluate_batch_parallel(new_candidates)
        
        # D. Update Dataset
        train_X = torch.cat([train_X, new_candidates])
        train_Y = torch.cat([train_Y, new_Y])
        
        # Progress Report
        best_val = -train_Y.max().item()
        print(f"  Iteration Complete. Best Residual Stress so far: {best_val:.2f} MPa")

    # 3. Save the final results
    results = {
        'X': train_X.numpy(),
        'Y': -train_Y.numpy() 
    }
    np.save('bo_final_results.npy', results)
    print("\nBayesian Optimization Complete. Results saved to bo_final_results.npy")