# Experiment 5: Neural Network Layers in CUDA

## 1. Aim
To implement neural network layers using CUDA and compare CPU and GPU performance.

## 2. Hardware and Software Requirements
- GPU: NVIDIA GeForce RTX 5060 Ti
- CUDA Toolkit: 13.0
- Visual Studio 2022 Build Tools
- Programming Language: CUDA C++

## 3. Network Dimensions

| Layer | Dimensions |
|---|---|
| Input | 256 × 128 |
| Hidden | 256 × 64 |
| Output | 256 × 10 |

## 4. CUDA Kernels Implemented
- Matrix Multiplication + Bias
- ReLU Activation
- Sigmoid Activation

## 5. Compilation and Execution

Compile the program:

```cmd
nvcc -O2 neural_network.cu -o neural_network.exe
```

Run the program:

```cmd
neural_network.exe
```

## 6. Performance Results

| Measurement | Result |
|---|---:|
| CPU average time | 0.740635 ms |
| GPU kernel-only time | 0.023630399 ms |
| GPU end-to-end time | 0.027195 ms |
| Kernel-only speedup | 31.34x |
| End-to-end speedup | 27.23x |
| Maximum absolute difference | 0.000000060 |

## 7. Test Results

- Matrix Multiplication + Bias: PASSED
- ReLU: PASSED
- Sigmoid: PASSED
- CPU and GPU outputs: MATCH (within tolerance)

## 8. Conclusion

The neural network operations were successfully implemented using CUDA. All kernel tests passed, and CPU and GPU outputs matched within the specified tolerance. The GPU achieved faster execution than the CPU.
