import os
import torch
import numpy as np
import concurrent.futures
import multiprocessing
from scipy.stats import qmc

# BoTorch Imports
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf

# --- CONFIGURATION ---
SIM_DURATION = 34.1
PROPERTIES_DIR = "."
GEOM_FILE = "thinwall.k"
NUM_PARAMS = 10

GPU_MAPPING = [1, 2, 3] 
BATCH_SIZE = len(GPU_MAPPING)

def isolated_simulation_worker(task_index, global_eval_id, params_np):
    """
    Strictly sandboxed worker.
    """
    gpu_id = GPU_MAPPING[task_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import cupy as cp
    from laser_generator import LaserProfileGenerator
    from thermomechanical import run_coupled_simulation
    
    cp.cuda.Device(0).use()
    print(f"[Eval {global_eval_id}] Initializing on GPU {gpu_id}...")
    
    generator = LaserProfileGenerator(total_time=SIM_DURATION)
    generator.generate_profile(params_np)
    
    # Naming continuously from 0 to 60
    unique_zarr = f"stress_history_eval_{global_eval_id}.zarr"
    
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
        print(f"[!] Eval {global_eval_id} FAILED: {e}")
        max_stress, avg_stress, avg_ht, min_ht = 2000.0, 1500.0, 0.0, 0.0
    
    cp.get_default_memory_pool().free_all_blocks()
    return [-avg_stress, avg_ht, min_ht, max_stress]

def generate_initial_data(num_samples):
    print(f"\n--- GENERATING INITIAL DATASET ({num_samples} SAMPLES) ---")
    sampler = qmc.LatinHypercube(d=NUM_PARAMS)
    init_X_np = sampler.random(n=num_samples)
    results = [None] * num_samples
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_idx = {
            # Passing iteration 0 for the initial generation
            executor.submit(isolated_simulation_worker, i % BATCH_SIZE, i, init_X_np[i]): i 
            for i in range(num_samples)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            print(f"  [+] Initial Sample {idx+1}/{num_samples} completed.")

    return torch.tensor(init_X_np, dtype=torch.float64), torch.tensor(results, dtype=torch.float64)

def run_optimization(train_X, train_Y, total_iterations=20):
    bounds = torch.tensor([[0.0] * NUM_PARAMS, [1.0] * NUM_PARAMS], dtype=torch.float64)
    
    T_MIN = 15.0     
    S_MAX = 1200.0   
    
    # YOUR ORIGINAL CONSTRAINTS (Negative = Feasible)
    constraints = [
        lambda Z: T_MIN - Z[..., 2], 
        lambda Z: Z[..., 3] - S_MAX  
    ]
    
    ref_point = torch.tensor([-1500.0, 0.0], dtype=torch.float64)

    # Calculate starting iteration from existing data
    start_iter = len(train_X) // BATCH_SIZE

    for iteration in range(start_iter, total_iterations):
        print(f"\n{'='*40}")
        print(f" BO ITERATION {iteration + 1} / {total_iterations}")
        print(f"{'='*40}")
        
        print("Training GP Models...")
        models = []
        for i in range(4):
                # Standardize Objectives so qEHVI isn't dominated by stress values
            models.append(SingleTaskGP(train_X, train_Y[..., i:i+1], outcome_transform=Standardize(m=1)))
                
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        # 1. Evaluate constraint feasibility on the historical data (Negative = Feasible)
        c1_hist = T_MIN - train_Y[..., 2]
        c2_hist = train_Y[..., 3] - S_MAX
        is_feasible = (c1_hist <= 0.0) & (c2_hist <= 0.0)
        
        # 2. Extract only the simulations that physically survived
        feasible_Y = train_Y[is_feasible]
        
        # 3. Build the Pareto front ONLY from feasible objectives
        if feasible_Y.shape[0] > 0:
            obj_Y_feasible = feasible_Y[..., 0:2]
            # Dynamic Reference Point: 10% worse than the worst feasible objectives
            ref_point = obj_Y_feasible.min(dim=0).values - (obj_Y_feasible.min(dim=0).values.abs() * 0.1)
        else:
            obj_Y_feasible = torch.empty((0, 2), dtype=torch.float64)
            # Fallback Reference Point if no feasible points exist yet
            ref_point = torch.tensor([-1500.0, 0.0], dtype=torch.float64)
            
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=obj_Y_feasible)
        
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            constraints=constraints,
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1])
        )
        
        print(f"Solving Acquisition Function for {BATCH_SIZE} candidates...")
        candidates, _ = optimize_acqf(
            acq_function=acq_func, bounds=bounds, q=BATCH_SIZE, 
            num_restarts=10, raw_samples=512, sequential=False
        )
        
        print(f"Dispatching physics simulations to GPUs...")
        new_results = [None] * BATCH_SIZE
        candidates_np = candidates.detach().numpy()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_idx = {}
            for i in range(BATCH_SIZE):
                global_eval_id = len(train_X) + i  # Unique ID for this evaluation, based on total evaluations so far
                # Passing the current iteration
                future = executor.submit(isolated_simulation_worker, i, global_eval_id , candidates_np[i])
                future_to_idx[future] = i

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
    multiprocessing.set_start_method('spawn', force=True)
    
    # Change this string to point to your existing .pt file
    CHECKPOINT_FILE = "bo_checkpoint_iter_1.pt" 
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n--- WARM START: RESUMING FROM {CHECKPOINT_FILE} ---")
        checkpoint = torch.load(CHECKPOINT_FILE)
        init_X = checkpoint['train_X']
        init_Y = checkpoint['train_Y']
        print(f"Loaded {len(init_X)} previous evaluations.")
    else:
        print(f"\n--- NO CHECKPOINT FOUND. COLD STARTING ---")
        init_X, init_Y = generate_initial_data(num_samples=BATCH_SIZE)
        
    # Run the optimization loop
    run_optimization(init_X, init_Y, total_iterations=30)