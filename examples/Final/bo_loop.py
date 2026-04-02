import os
import torch
import numpy as np
import concurrent.futures
import multiprocessing
import warnings
import logging
import glob
import re
from scipy.stats import qmc
import gpytorch

# BoTorch Imports
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize 
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
from botorch.optim.optimize import optimize_acqf_mixed  
from botorch.exceptions.warnings import InputDataWarning
from gpytorch.kernels import RBFKernel, ScaleKernel


# Suppress the zero-variance warning that floods the terminal during early iterations
warnings.filterwarnings("ignore", category=InputDataWarning)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bo_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SIM_DURATION = 27.53571430
PROPERTIES_DIR = "."
GEOM_FILE = "wall.k"
NUM_PARAMS = 10

GPU_MAPPING = [1, 2, 3] 
BATCH_SIZE = len(GPU_MAPPING)

def isolated_simulation_worker(task_index, global_eval_id, params_np):
    gpu_id = GPU_MAPPING[task_index]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        
        from laser_generator import LaserProfileGenerator
        from thermomechanical import run_coupled_simulation
        
        logger.info(f"[Eval {global_eval_id}] Initializing on GPU {gpu_id}...")
        
        generator = LaserProfileGenerator(total_time=SIM_DURATION)
        generator.generate_profile(params_np)
        
        unique_zarr = f"stress_history_eval_{global_eval_id}.zarr"
        
        # Unpack the 6 returned values
        max_stress, avg_stress, avg_ht, min_ht, max_warpage, max_peeq = run_coupled_simulation(
            params=params_np,
            generator=generator,
            input_dir=PROPERTIES_DIR,
            geom_file=GEOM_FILE,
            output_path=unique_zarr,
            active_print_time=SIM_DURATION
        )
        cp.get_default_memory_pool().free_all_blocks()
        
    except Exception as e:
        logger.error(f"[!] Eval {global_eval_id} FAILED on GPU {gpu_id}: {e}")
        import numpy as np
        jitter = np.random.uniform(0, 1e-3)
        max_stress = 2000.0 + jitter
        avg_stress = 1500.0 + jitter
        avg_ht = 0.0 + jitter
        min_ht = 0.0 + jitter
        max_warpage = 9999.0 + jitter
        max_peeq = 9999.0 + jitter
        
    return [-avg_stress, avg_ht, min_ht, max_stress, max_warpage, max_peeq]

def generate_initial_data(num_samples):
    logger.info(f"\n--- GENERATING INITIAL DATASET ({num_samples} SAMPLES) ---")
    sampler = qmc.LatinHypercube(d=NUM_PARAMS, optimization="random-cd") 
    init_X_np = sampler.random(n=num_samples)
    
    init_X_np[:, 0] = np.round(init_X_np[:, 0] * 10) / 10.0 
    
    results = [None] * num_samples
    
    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, num_samples)
        current_batch_size = batch_end - batch_start
        
        logger.info(f"\nStarting Initial LHS Batch: Samples {batch_start+1} to {batch_end}")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=current_batch_size) as executor:
            future_to_idx = {}
            for local_i in range(current_batch_size):
                global_i = batch_start + local_i
                future = executor.submit(isolated_simulation_worker, local_i, global_i, init_X_np[global_i])
                future_to_idx[future] = global_i
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    logger.info(f"  [+] Initial Sample {idx+1}/{num_samples} completed.")
                except Exception as pool_error:
                    logger.error(f"  [!] FATAL POOL CRASH: {pool_error}")
                    jitter = np.random.uniform(0, 1e-3)
                    results[idx] = [-1500.0 - jitter, 0.0 + jitter, 0.0 + jitter, 2000.0 + jitter, 9999.0 + jitter, 9999.0 + jitter]
    return torch.tensor(init_X_np, dtype=torch.float64), torch.tensor(results, dtype=torch.float64)

def run_optimization(train_X, train_Y, train_acq_vals, total_iterations=20):
    # --- GPU 0 ROUTING ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Executing Bayesian Optimization Math on: {device}")
    
    # Push historical data and static bounds to the GPU
    train_X = train_X.to(device=device, dtype=torch.float64)
    train_Y = train_Y.to(device=device, dtype=torch.float64)
    if train_acq_vals.nelement() > 0:
        train_acq_vals = train_acq_vals.to(device=device)
        
    bounds = torch.tensor([[0.0] * NUM_PARAMS, [1.0] * NUM_PARAMS], dtype=torch.float64, device=device)
    SCALED_REF_POINT = torch.tensor([-0.05, -0.05], dtype=torch.float64, device=device)
    
    STRICT_S_MAX = 320.0
    STRICT_T_MIN = 0.05
    
    EXPLORE_S_MAX = STRICT_S_MAX * 1.02
    EXPLORE_T_MIN = STRICT_T_MIN * 0.98
       
    bo_batches_run = max(0, (len(train_X) - 51) // BATCH_SIZE)
    start_iter = bo_batches_run

    for iteration in range(start_iter, total_iterations):
        logger.info(f"\n{'='*40}")
        logger.info(f" BO ITERATION {iteration + 1} / {total_iterations}")
        logger.info(f"{'='*40}")
        
        logger.info("Training 4 Independent SingleTaskGPs...")
        models = []
        for i in range(4):
            covar = ScaleKernel(RBFKernel(ard_num_dims=NUM_PARAMS))
            gp = SingleTaskGP(
                train_X, 
                train_Y[..., i:i+1], 
                covar_module=covar, 
                input_transform=Normalize(d=NUM_PARAMS),
                outcome_transform=Standardize(m=1) 
            )
            models.append(gp)
            
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        try:
            with gpytorch.settings.cholesky_jitter(1e-6):
                fit_gpytorch_mll(mll)  
        except Exception as e:
            logger.error(f"GP Fitting Failed: {e}. Attempting to continue.")

        constraints = [
            lambda Z: Z[..., 3] - EXPLORE_S_MAX, 
            lambda Z: EXPLORE_T_MIN - Z[..., 2]   
        ]

        Y_obj_all = train_Y[..., 0:2]
        Y_min = Y_obj_all.min(dim=0).values
        Y_max = Y_obj_all.max(dim=0).values
        Y_range = (Y_max - Y_min).clamp(min=1e-6)
        
        def scaled_objective(Z, X=None):
            return (Z[..., 0:2] - Y_min) / Y_range
        
        c1_hist = STRICT_T_MIN - train_Y[..., 2] 
        c2_hist = train_Y[..., 3] - STRICT_S_MAX
        is_feasible = (c2_hist <= 0.0) & (c1_hist <= 0.0)
        
        feasible_Y = train_Y[is_feasible]
        if feasible_Y.shape[0] > 0:
            scaled_Y_feasible = (feasible_Y[..., 0:2] - Y_min) / Y_range
        else:
            scaled_Y_feasible = torch.empty((0, 2), dtype=torch.float64, device=device) # Initialized on device

        partitioning = NondominatedPartitioning(ref_point=SCALED_REF_POINT, Y=scaled_Y_feasible)
        
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=SCALED_REF_POINT,
            partitioning=partitioning,
            constraints=constraints,
            objective=GenericMCMultiOutputObjective(scaled_objective)
        )
        
        logger.info(f"Solving Mixed Acquisition Function for {BATCH_SIZE} candidates...")
        
        n_choices = [i / 10.0 for i in range(11)]
        fixed_features_list = [{0: choice} for choice in n_choices]

        candidates, acq_value = optimize_acqf_mixed(
            acq_function=acq_func, 
            bounds=bounds, 
            q=BATCH_SIZE, 
            num_restarts=40, 
            raw_samples=2**12,
            fixed_features_list=fixed_features_list
        )
        
        logger.info(f"  -> Acquisition Function Value (qLogEHVI): {acq_value.item():.4f}")
        logger.info(f"Dispatching physics simulations to GPUs...")
        
        new_results = [None] * BATCH_SIZE
        
        # --- PULL TO CPU FOR MULTIPROCESSING ---
        candidates_np = candidates.detach().cpu().numpy()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=BATCH_SIZE) as executor:
            future_to_idx = {}
            for i in range(BATCH_SIZE):
                global_eval_id = len(train_X) + i 
                future = executor.submit(isolated_simulation_worker, i, global_eval_id, candidates_np[i])
                future_to_idx[future] = i

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    new_results[idx] = future.result()
                    logger.info(f"  [+] BO Task {idx:02d} (GPU {GPU_MAPPING[idx]}) finished.")
                except Exception as pool_error:
                    logger.error(f"  [!] FATAL POOL CRASH on Task {idx:02d}: {pool_error}")
                    jitter = np.random.uniform(0, 1e-3)
                    new_results[idx] = [-1500.0 - jitter, 0.0 + jitter, 0.0 + jitter, 2000.0 + jitter, 9999.0 + jitter, 9999.0 + jitter]

        # --- PUSH NEW RESULTS TO GPU 0 ---
        new_Y = torch.tensor(new_results, dtype=torch.float64, device=device)
        train_X = torch.cat([train_X, candidates]) # Candidates is already on GPU 0
        train_Y = torch.cat([train_Y, new_Y])
        
        if train_acq_vals.dim() == 0 or train_acq_vals.nelement() == 0:
            train_acq_vals = acq_value.unsqueeze(0)
        else:
            train_acq_vals = torch.cat([train_acq_vals, acq_value.unsqueeze(0)])
        
        # --- PULL TO CPU FOR SAFE SAVING ---
        torch.save({
            'train_X': train_X.cpu(), 
            'train_Y': train_Y.cpu(),
            'train_acq_vals': train_acq_vals.cpu()
        }, f"bo_checkpoint_iter_{iteration+1}.pt")
        
        logger.info(f"Iteration complete. Data saved.")

    return train_X, train_Y

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    checkpoint_files = glob.glob("bo_checkpoint_iter_*.pt")
    latest_checkpoint = None
    max_iter = -1
    
    for file in checkpoint_files:
        match = re.search(r'iter_(\d+)\.pt', file)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
                latest_checkpoint = file

    if latest_checkpoint:
        logger.info(f"\n--- WARM START: RESUMING FROM {latest_checkpoint} ---")
        checkpoint = torch.load(latest_checkpoint)
        init_X = checkpoint['train_X']
        init_Y = checkpoint['train_Y']
        init_acq = checkpoint.get('train_acq_vals', torch.empty(0)) 
        logger.info(f"Loaded {len(init_X)} previous evaluations.")
    else:
        logger.info(f"\n--- NO CHECKPOINT FOUND. COLD STARTING ---")
        init_X, init_Y = generate_initial_data(num_samples=51)
        init_acq = torch.empty(0) 
        
        torch.save({
            'train_X': init_X, 
            'train_Y': init_Y,
            'train_acq_vals': init_acq
        }, "bo_checkpoint_iter_0.pt")
        logger.info("Saved Initial LHS dataset to bo_checkpoint_iter_0.pt")
        
    run_optimization(init_X, init_Y, init_acq, total_iterations=50)