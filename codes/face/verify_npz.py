import numpy as np
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'data/FLAME_albedo_from_BFM.npz'

if not os.path.exists(path):
    print(f"ERROR: File does not exist: {path}")
    sys.exit(1)

size = os.path.getsize(path)
print(f"File size: {size} bytes")

if size < 1000000:
    print(f"ERROR: File too small ({size} bytes) - likely an HTML error page")
    sys.exit(1)

try:
    data = np.load(path, allow_pickle=True)
    print("Keys:", list(data.keys()))
    print("VALID NPZ FILE - All good!")
except Exception as e:
    print(f"ERROR loading npz: {e}")
    sys.exit(1)
