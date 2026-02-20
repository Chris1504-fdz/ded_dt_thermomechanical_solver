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
        Pre-calculates the entire power profile to establish global min/max 
        values for accurate normalization. Must be called before simulation.
        """
        # Create full time array
        self.time_array = np.arange(0, self.total_time, self.time_step)
        x_norm = self._normalize_time(self.time_array)

        # Unpack params
        n, freq, amplitude, phase, trend, seasonality, frequency_slope, amplitude_slope, phase_slope, seasonality_freq = params
        
        # Rescale params based on designated bounds
        n = int(self._rescale(n, 0, 10))
        if n < 1: n = 1 
        
        freq = self._rescale(freq, 0, 10)
        amplitude = self._rescale(amplitude, 0, 10)
        phase = self._rescale(phase, 0, 10000)
        trend = self._rescale(trend, -500, 500)
        seasonality = self._rescale(seasonality, 0, 500)
        frequency_slope = self._rescale(frequency_slope, -1.25, 1.25)
        amplitude_slope = self._rescale(amplitude_slope, -1.25, 1.25)
        phase_slope = self._rescale(phase_slope, -1.25, 1.25)
        seasonality_freq = self._rescale(seasonality_freq, -1, 1)

        # 1. Base Fourier Series
        sum_val = np.zeros_like(x_norm)
        for i in range(1, n + 1, 2):
            term = (1 / i) * np.sin(2 * np.pi * (freq + i * frequency_slope) * i * x_norm + (phase + i * phase_slope))
            sum_val += term

        y = (amplitude + n * amplitude_slope) * (2 / np.pi) * sum_val
        
        # 2. Intermediate Normalization
        if np.max(y) != np.min(y):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            
        y = (y * 50) + 600  # rescale_amplitude=50, rescale_mag=600

        # 3. Apply Trend and Seasonality
        y += trend * x_norm
        y += seasonality * np.sin(2 * np.pi * seasonality_freq * x_norm)

        # 4. Final Global Normalization to Laser Limits
        if np.max(y) == np.min(y):
            # Fallback for a perfectly flat signal
            self.power_array = np.full_like(y, 600) 
        else:
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
            self.power_array = self._rescale(y_norm, min_power, max_power)

    def get_power_at_time(self, t):
        """
        Fetches the power at a specific scalar time 't' using linear interpolation.
        """
        if self.time_array is None or self.power_array is None:
            raise RuntimeError("You must call generate_profile(params) before fetching power.")
        
        # np.interp handles scalar 't' perfectly, even if 't' falls between exact time_steps
        return np.interp(t, self.time_array, self.power_array)