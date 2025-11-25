import torch
import os

# config
NUM_X_BLOCKS = int(20E3)
THREAD_1D_LEN = 32
TOTAL_SIZE = (THREAD_1D_LEN**2)
ITER = int(16E3)

# setup
devc = torch.device(type="cuda")
dtype = torch.float32
torch.set_printoptions(sci_mode=False)
os.system("cls")
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
gconf = {
    "dtype": dtype,
    "device": devc
}
size3d = [NUM_X_BLOCKS, THREAD_1D_LEN, THREAD_1D_LEN] # size [20000, 32, 32]
size2d = [THREAD_1D_LEN, THREAD_1D_LEN] # size [32, 32]

# def kernels
def proc(A, B, C, G):
    G = (A @ B) + C

# prefill -- domain=[0, 1)
print(f"prefilling...")
"""
A needs to be a 3d tensor was due to the kernel caching results and not actually
processing all the data, thus A will contain random data to force it to process thoroughly
"""
A = torch.rand(size3d, **gconf)
B = torch.rand(size2d, **gconf)
C = torch.rand(size2d, **gconf)
G = torch.zeros(size3d, **gconf)

# timer start
torch.cuda.synchronize()
start_event.record()

# process
print(f"processing...")
for _ in range(ITER):
    proc(A, B, C, G)

# timer end
end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"took: {elapsed_time_ms:.2f}ms")

# DEBUG
# print(A, end="\n\n")
# print(B, end="\n\n")
# print(C, end="\n\n")
# print(G, end="\n\n")