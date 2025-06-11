#!/usr/bin/env python3

try :  
    import torch 
except ImportError :
    print("torch not found!")

def get_device() -> torch.device:
    """
    this function will try to find either a cuda device or mac metal 
    """
    if torch.cuda.is_available() :
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

if __name__ == "__main__" :
    print(f"found device {get_device()}")
