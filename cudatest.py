import torch
if torch.cuda.is_available():
    print("Cuda available.")
    exit(0)
else:
    print("Cuda not available.")
    exit(1)
