import os
import torch

print("USE_LIBUV:", os.getenv("USE_LIBUV", "Not set"))
print("Torch version:", torch.__version__)