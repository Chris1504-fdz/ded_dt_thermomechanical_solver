import numpy as np

class LaserProfileGenerator:
    def __init__(self, total_time=300, time_step=0.002):
        self.total_time = total_time
        self.time_step = time_step
        
        # Placeholders for the pre-calculated arrays
        self.time_array = None
        self.power_array = None

    def _normalize_time(self, x):
        """
        Normalize the time array to [-1, 1] based on total simulation duration.
        """
        return 2 * (x / self.total_time) - 1

    def _rescale(self, x, min_value, max_value):
        """Rescale a normalized input [0, 1] to physical range [min, max]."""
        return x * (max_value - min_value) + min_value

    def generate_profile(self, params, min_power=400, max_power=700):
        """
        Maps normalized BO parameters to physical laser outputs.
        """
        self.time_array = np.arange(0, self.total_time, self.time_step)
        x_norm = self._normalize_time(self.time_array)

        # Unpack params
        n, freq, amplitude, phase, trend, seasonality, frequency_slope, amplitude_slope, phase_slope, seasonality_freq = params
        
        # 1. Map to physical constraints
        # We MUST cast 'n' to an integer so the range() function doesn't crash
        n_int = int(np.round(self._rescale(n, 0, 10)))
        
        freq = self._rescale(freq, 0.1, 10.0)
        amplitude_watts = self._rescale(amplitude, 0, 150) # Actual physical Watts!
        phase = self._rescale(phase, 0, 2 * np.pi)
        
        trend_watts = self._rescale(trend, -150, 150)
        seasonality_watts = self._rescale(seasonality, 0, 100)
        
        frequency_slope = self._rescale(frequency_slope, -1.0, 1.0)
        amplitude_slope = self._rescale(amplitude_slope, -1.0, 1.0)
        phase_slope = self._rescale(phase_slope, -1.0, 1.0)
        seasonality_freq = self._rescale(seasonality_freq, 0.5, 5.0)

        # 2. Base Fourier Series (Original discrete formulation)
        sum_val = np.zeros_like(x_norm)
        # Added +1 so if n_int=5, it actually includes the 5th harmonic (1, 3, 5)
        for i in range(1, n_int + 1, 2):
            term_weight = 1 / i
            term = term_weight * np.sin(2 * np.pi * (freq + i * frequency_slope) * i * x_norm + (phase + i * phase_slope))
            sum_val += term

        baseline_power = (self._rescale(0.5, min_power, max_power)) 
        
        # 3. Assemble Physical Profile
        if np.max(sum_val) != np.min(sum_val):
            sum_val = sum_val / np.max(np.abs(sum_val)) 
            
        envelope = amplitude_watts + (x_norm * amplitude_slope * 50)
        envelope = np.maximum(envelope, 0.0)    
        y = baseline_power + (sum_val * envelope) 
        
        # 4. Add Macroscopic Drifts
        y += trend_watts * x_norm
        y += seasonality_watts * np.sin(2 * np.pi * seasonality_freq * x_norm)

        # 5. Hardware Constraint Squashing
        y = baseline_power + (max_power - baseline_power) * np.tanh((y - baseline_power) / (max_power - baseline_power))
        self.power_array = y
        
    def get_power_at_time(self, t):
        """
        Fetches the power at a specific scalar time 't' using linear interpolation.
        """
        if self.time_array is None or self.power_array is None:
            raise RuntimeError("You must call generate_profile(params) before fetching power.")
        
        # np.interp handles scalar 't' perfectly, even if 't' falls between exact time_steps
        return np.interp(t, self.time_array, self.power_array)