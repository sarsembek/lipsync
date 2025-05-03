import torch
from typing import Literal

def get_device(verbose: bool = True) -> torch.device:
    """
    Automatically selects the best available device (CUDA > MPS > CPU).
    
    Args:
        verbose: Whether to print the selected device.
        
    Returns:
        torch.device: The selected computing device.
    """
    device_priority: list[tuple[Literal["cuda", "mps", "cpu"], str]] = [
        ("cuda", "NVIDIA GPU"),
        ("mps", "Apple Silicon (MPS)"),
        ("cpu", "CPU")
    ]
    
    for device_type, device_name in device_priority:
        try:
            if device_type == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
                if verbose:
                    print(f"🚀 Using {device_name} ({torch.cuda.get_device_name(0)})")
                return device
            elif device_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                if verbose:
                    print(f"🍏 Using {device_name} (MPS)")
                return device
            elif device_type == "cpu":
                if verbose:
                    print("🖥️  Using CPU")
                return torch.device("cpu")
        except Exception as e:
            if verbose:
                print(f"⚠️  Error initializing {device_name}: {str(e)}")
    
    if verbose:
        print("⚠️  Falling back to CPU")
    return torch.device("cpu")
