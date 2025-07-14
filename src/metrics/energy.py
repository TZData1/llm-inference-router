import time
import numpy as np
import pynvml
import threading
import logging
from abc import ABC, abstractmethod

try:
    from zeus.monitor import ZeusMonitor
    ZEUS_AVAILABLE = True



except ImportError:
    print("Zeus is not available. Install with: pip install zeus-ml")
    ZEUS_AVAILABLE = False

class BasePowerMeasurement(ABC):
    """Abstract base class for power measurement"""
    
    @abstractmethod
    def start(self):
        """Start power measurement"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop measurement and return energy used (joules)"""
        pass

class PynvmlMeasurement(BasePowerMeasurement):
    """Energy measurement using NVIDIA Management Library."""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.start_time = None
        self.power_samples = []
        self.sampling_interval = 0.1
        self._initialize_nvml()
        
    def _initialize_nvml(self):
        """Initialize NVML library."""
        try:
            pynvml.nvmlInit()
            self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            print(f"NVML initialized for device {self.device_id}")
        except Exception as e:
            print(f"NVML initialization failed: {e}")
            self.device_handle = None
            
    def start(self):
        """Start energy measurement."""
        self.start_time = time.time()
        self.power_samples = []
        self.stop_measurement = False
        
        if self.device_handle:
            self.measurement_thread = threading.Thread(target=self._sample_power)
            self.measurement_thread.daemon = True
            self.measurement_thread.start()
        
        return True
    
    def _sample_power(self):
        """Sample power in background thread."""
        while not self.stop_measurement:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0
                self.power_samples.append(power)
            except Exception:
                pass
                
            time.sleep(self.sampling_interval)
    
    def stop(self):
        """Stop measurement and return energy used (joules)."""
        if self.start_time is None:
            return 0
        
        duration = time.time() - self.start_time
        self.stop_measurement = True
        if self.device_handle and self.power_samples:
            if hasattr(self, 'measurement_thread'):
                self.measurement_thread.join(timeout=0.5)
            mean_power = np.mean(self.power_samples)
            energy = mean_power * duration
            return energy
        else:
            raise RuntimeError("GPU power measurement failed: NVML device handle unavailable and no power samples collected")

class ZeusMeasurement(BasePowerMeasurement):
    """GPU energy measurement using Zeus Python API."""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.monitor = None
        self.window_name = f"inference_{time.time()}"
        self.start_time = None
        
        if not ZEUS_AVAILABLE:
            print("Zeus is not available. Install with: pip install zeus-ml")
            
    def start(self):
        """Start Zeus power measurement for GPU."""
        self.start_time = time.time()
        
        if not ZEUS_AVAILABLE:
            return False
            
        try:
            self.monitor = ZeusMonitor(gpu_indices=[self.device_id])
            self.monitor.begin_window(self.window_name)
            return True
        except Exception as e:
            print(f"Zeus start failed: {e}")
            self.monitor = None
            return False
    
    def stop(self):
        """Stop Zeus measurement and return GPU energy used (joules)."""
        if self.start_time is None or self.monitor is None:
            raise RuntimeError("Zeus energy measurement failed: monitor not initialized or start time missing")
        
        try:
            measurement = self.monitor.end_window(self.window_name)
            return measurement.total_energy
        except Exception as e:
            raise RuntimeError(f"Zeus energy measurement failed during stop: {e}")

class EnergyMeasurement:
    """Factory class to create the appropriate energy measurement tool."""
    
    @staticmethod
    def create(method="zeus", device_id=0, **kwargs):
        """Create energy measurement instance based on method."""
        if method == "zeus" and ZEUS_AVAILABLE:
            print("Create energy measurement instance Zeus.")
            return ZeusMeasurement(device_id=device_id, **kwargs)
        else:
            print("Create energy measurement instance PyNVML.")
            return PynvmlMeasurement(device_id=device_id)

class EnergyMetrics:
    """Calculate energy-related metrics."""
    
    def calculate_per_token(self, total_energy, input_tokens, output_tokens):
        """Calculate energy per token."""
        total_tokens = input_tokens + output_tokens
        if total_tokens == 0:
            return 0.0
            
        return total_energy / total_tokens