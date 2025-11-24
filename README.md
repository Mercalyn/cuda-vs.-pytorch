# Performance Cuda vs. Pytorch
Methodology
Using this PC setup:
- HOST: AMD Ryzen 5 3600, 32GB System Mem, MSI B450M Gaming Plus
- GPU: Nvidia RTX 3090 Ti 24GB VRAM

Similar tests will be performed in Cuda and Pytorch in order to witness how much slower Pytorch is compared to a raw Cuda kernel.
The test:
A, B, C, will be 2d arrays of 32x32
G will be a 3d array that can collect the full result

16,000 iterations of:
  20,000 blocks of
    matrix multiplications temp = A @ B
    setting G = temp + C

The iterations serve as a sequential loader, and to visually confirm that the GPU load hits 100%.
The 20,000 block matrix multiplications and bias adders serve as a parallel loader



## ------------------------------- CUDA ------------------------------- 
### Processing Results:
9115ms
9101ms
9101ms
9121ms
9105ms


### Interesting cuda compiler caching speedups
A)
When the processing kernel does not feed to G (produces no output), a speedup from 9100ms to 615ms is observed. With no output actually being produced, the cuda compiler is taking a smart shortcut somewhere. Likely the kernel runs but is not performing any multiplications.
B)
Normally within one single multiplication it must select the correct location of A and B to multiply-accumulate. If this is replaced with a fixed multiplication of .5 * .5 and then accumulated, there is some smart caching somewhere and a speedup from 9100ms to 1440ms is observed. This test contains an empty lookup for A and B in order to prove the speedup does not come from the lack of data lookups (in code), but caching of data.


### Non-optimized thread size
Thread size vs. memory useage matters but is also a pain to customize. Only a (32, 32) block size of threads was tested. Performance will reflect an unoptimized cuda kernel.



## ------------------------------- PYTORCH ------------------------------- 
Processing Results:
-
-
-
-
-


Interpretation of Cuda blocks will exist in Pytorch's Z dimension. In other words, Cuda's 20,000 blocks in the X direction will be translated to a depth dimension in Pytorch, giving a size of [20000, 32, 32], allowing this portion to also run in parallel.
