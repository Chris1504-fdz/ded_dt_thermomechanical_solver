
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import numpy as np
import cupy as cp
import time

# Import your custom modules
from laser_generator import LaserProfileGenerator
from thermomechanical import run_coupled_simulation

# --- CONFIGURATION ---
SIM_DURATION = 121.1  # Ensure this matches your .crs file duration
PROPERTIES_DIR = "."
GEOM_FILE = "thinwall.k"
OUTPUT_ZARR = "debug_stress.zarr"
cp.cuda.Device(0).use()

def run_debug():
    print("--- Starting Debug Simulation ---")
    
    # 1. Generate 10 random parameters (range [0, 1])
    random_params = np.random.rand(10)
    print(f"Random Parameters: \n{random_params}\n")

    # 2. Initialize the Laser Generator
    generator = LaserProfileGenerator(total_time=SIM_DURATION)
    
    # ---> CRITICAL FIX: Pre-calculate the profile before trying to preview it
    generator.generate_profile(random_params)
    
    # 3. Optional: Preview the generated power curve (first few steps)
    print("Previewing power profile...")
    test_times = [0.0, SIM_DURATION * 0.25, SIM_DURATION * 0.5, SIM_DURATION * 0.75]
    for t in test_times:
        # ---> CRITICAL FIX: Remove 'random_params' from this call
        p = generator.get_power_at_time(t)
        print(f"  t={t:.3f}s -> Power={p:.2f}W")
    print("")

    # 4. Run the Coupled Simulation
    start_wall = time.perf_counter()
    
    try:
        print(f"Executing Thermomechanical Solver on GPU {cp.cuda.Device().id}...")
        max_residual_stress = run_coupled_simulation(
            params=random_params,
            generator=generator,
            input_dir=PROPERTIES_DIR,
            geom_file=GEOM_FILE,
            output_path=OUTPUT_ZARR
        )
        
        end_wall = time.perf_counter()
        
        print("\n--- Simulation Success ---")
        print(f"Wall Time: {end_wall - start_wall:.2f} seconds")
        print(f"Max Residual Stress: {max_residual_stress:.2f} MPa")
        print(f"Results saved to: {os.path.abspath(OUTPUT_ZARR)}")
        
    except Exception as e:
        print(f"\n!!! Simulation Failed !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()