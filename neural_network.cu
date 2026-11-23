
// EXPERIMENT 5: Neural Network Layers in CUDA
// Matrix multiplication + bias, ReLU, and Sigmoid
// CPU and GPU comparison

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int BATCH = 256;
constexpr int INPUT_SIZE = 128;
constexpr int HIDDEN_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;
constexpr int REPEATS = 20;

// PHASE 2: CUDA KERNELS

__global__ void matrixMulBiasKernel(
    const float* A, const float* B, const float* bias, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum + bias[col];
    }
}

__global__ void reluKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = fmaxf(0.0f, values[idx]);
    }
}

__global__ void sigmoidKernel(float* values, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] = 1.0f / (1.0f + expf(-values[idx]));
    }
}

// PHASE 3: CPU IMPLEMENTATION

void cpuMatMulBias(const std::vector<float>& A,
                   const std::vector<float>& B,
                   const std::vector<float>& bias,
                   std::vector<float>& C,
                   int M, int N, int K)
{
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum + bias[col];
        }
    }
}

void cpuReLU(std::vector<float>& values)
{
    for (float& x : values) {
        x = std::max(0.0f, x);
    }
}

void cpuSigmoid(std::vector<float>& values)
{
    for (float& x : values) {
        x = 1.0f / (1.0f + std::exp(-x));
    }
}

void cpuForwardPass(const std::vector<float>& X,
                    const std::vector<float>& W1,
                    const std::vector<float>& b1,
                    const std::vector<float>& W2,
                    const std::vector<float>& b2,
                    std::vector<float>& output)
{
    std::vector<float> z1(BATCH * HIDDEN_SIZE);

    cpuMatMulBias(X, W1, b1, z1,
                  BATCH, HIDDEN_SIZE, INPUT_SIZE);
    cpuReLU(z1);

    std::vector<float> z2(BATCH * OUTPUT_SIZE);

    cpuMatMulBias(z1, W2, b2, z2,
                  BATCH, OUTPUT_SIZE, HIDDEN_SIZE);
    cpuSigmoid(z2);

    output = z2;
}

// PHASE 4: GPU FORWARD PASS

void launchGpuForwardPass(
    const float* d_X,
    const float* d_W1,
    const float* d_b1,
    const float* d_W2,
    const float* d_b2,
    float* d_z1,
    float* d_z2)
{
    dim3 block(16, 16);

    dim3 grid1(
        (HIDDEN_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    matrixMulBiasKernel<<<grid1, block>>>(
        d_X, d_W1, d_b1, d_z1,
        BATCH, HIDDEN_SIZE, INPUT_SIZE
    );

    CUDA_CHECK(cudaGetLastError());

    int count1 = BATCH * HIDDEN_SIZE;
    reluKernel<<<(count1 + 255) / 256, 256>>>(d_z1, count1);

    CUDA_CHECK(cudaGetLastError());

    dim3 grid2(
        (OUTPUT_SIZE + 15) / 16,
        (BATCH + 15) / 16
    );

    matrixMulBiasKernel<<<grid2, block>>>(
        d_z1, d_W2, d_b2, d_z2,
        BATCH, OUTPUT_SIZE, HIDDEN_SIZE
    );

    CUDA_CHECK(cudaGetLastError());

    int count2 = BATCH * OUTPUT_SIZE;
    sigmoidKernel<<<(count2 + 255) / 256, 256>>>(d_z2, count2);

    CUDA_CHECK(cudaGetLastError());
}

bool nearlyEqual(float a, float b, float tolerance = 1e-4f)
{
    return std::fabs(a - b) <= tolerance;
}

// PHASE 5: INDIVIDUAL KERNEL TESTS

bool testMatrixKernel()
{
    const int M = 2, N = 3, K = 2;

    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8, 9, 10};
    std::vector<float> bias = {1, 2, 3};

    std::vector<float> expected(M * N);
    std::vector<float> actual(M * N);

    cpuMatMulBias(A, B, bias, expected, M, N, K);

    float *d_A, *d_B, *d_bias, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, A.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, B.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, actual.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(1, 1);

    matrixMulBiasKernel<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(actual.data(), d_C, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;

    for (int i = 0; i < M * N; ++i) {
        if (!nearlyEqual(expected[i], actual[i])) {
            passed = false;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_bias);
    cudaFree(d_C);

    return passed;
}

bool testReLUKernel()
{
    std::vector<float> values = {-2, -1, 0, 1, 2};
    std::vector<float> expected = {0, 0, 0, 1, 2};

    float* d_values;

    CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

    reluKernel<<<1, 256>>>(d_values, (int)values.size());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(values.data(), d_values, values.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;

    for (size_t i = 0; i < values.size(); ++i) {
        if (!nearlyEqual(values[i], expected[i])) {
            passed = false;
        }
    }

    cudaFree(d_values);
    return passed;
}

bool testSigmoidKernel()
{
    std::vector<float> values = {-2, -1, 0, 1, 2};
    std::vector<float> expected(5);

    for (int i = 0; i < 5; ++i) {
        expected[i] = 1.0f / (1.0f + std::exp(-values[i]));
    }

    float* d_values;

    CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

    sigmoidKernel<<<1, 256>>>(d_values, (int)values.size());

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(values.data(), d_values, values.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool passed = true;

    for (size_t i = 0; i < values.size(); ++i) {
        if (!nearlyEqual(values[i], expected[i])) {
            passed = false;
        }
    }

    cudaFree(d_values);
    return passed;
}

// PHASE 6: MAIN PROGRAM AND CPU EXECUTION

int main()
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA GPU found!" << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "========================================\n";
    std::cout << " EXPERIMENT 5: NEURAL NETWORK CUDA\n";
    std::cout << "========================================\n";
    std::cout << "GPU: " << prop.name << "\n\n";

    std::cout << "Kernel Tests:\n";

    bool matrixPassed = testMatrixKernel();
    bool reluPassed = testReLUKernel();
    bool sigmoidPassed = testSigmoidKernel();

    std::cout << "Matrix Multiplication + Bias: "
              << (matrixPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "ReLU: "
              << (reluPassed ? "PASSED" : "FAILED") << "\n";

    std::cout << "Sigmoid: "
              << (sigmoidPassed ? "PASSED" : "FAILED") << "\n\n";

    if (!matrixPassed || !reluPassed || !sigmoidPassed) {
        std::cerr << "Kernel test failed!\n";
        return EXIT_FAILURE;
    }

    std::vector<float> X(BATCH * INPUT_SIZE);
    std::vector<float> W1(INPUT_SIZE * HIDDEN_SIZE);
    std::vector<float> b1(HIDDEN_SIZE);
    std::vector<float> W2(HIDDEN_SIZE * OUTPUT_SIZE);
    std::vector<float> b2(OUTPUT_SIZE);

    std::vector<float> cpuOutput(BATCH * OUTPUT_SIZE);
    std::vector<float> gpuOutput(BATCH * OUTPUT_SIZE);

    srand(42);

    auto randomValue = []() {
        return static_cast<float>(rand()) / RAND_MAX * 0.2f - 0.1f;
    };

    for (float& x : X) x = randomValue();
    for (float& x : W1) x = randomValue();
    for (float& x : b1) x = randomValue();
    for (float& x : W2) x = randomValue();
    for (float& x : b2) x = randomValue();

    std::cout << "Network Dimensions:\n";
    std::cout << "Input:  " << BATCH << " x " << INPUT_SIZE << "\n";
    std::cout << "Hidden: " << BATCH << " x " << HIDDEN_SIZE << "\n";
    std::cout << "Output: " << BATCH << " x " << OUTPUT_SIZE << "\n\n";

    auto cpuStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPEATS; ++i) {
        cpuForwardPass(X, W1, b1, W2, b2, cpuOutput);
    }

    auto cpuStop = std::chrono::high_resolution_clock::now();

    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuStop - cpuStart).count() / REPEATS;

    // PHASE 7: GPU MEMORY, EXECUTION AND TIMING

    float *d_X, *d_W1, *d_b1, *d_W2, *d_b2;
    float *d_z1, *d_z2;

    CUDA_CHECK(cudaMalloc(&d_X, X.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1, W1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, b1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2, W2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, b2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z1, BATCH * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_z2, BATCH * OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, X.data(), X.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W1, W1.data(), W1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), b1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, W2.data(), W2.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), b2.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up
    launchGpuForwardPass(d_X, d_W1, d_b1, d_W2, d_b2, d_z1, d_z2);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        launchGpuForwardPass(d_X, d_W1, d_b1, d_W2, d_b2, d_z1, d_z2);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuKernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuKernelMs, start, stop));
    gpuKernelMs /= REPEATS;

    auto gpuTotalStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < REPEATS; ++i) {
        launchGpuForwardPass(d_X, d_W1, d_b1, d_W2, d_b2, d_z1, d_z2);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gpuOutput.data(), d_z2,
                          gpuOutput.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto gpuTotalStop = std::chrono::high_resolution_clock::now();

    double gpuEndToEndMs =
        std::chrono::duration<double, std::milli>(
            gpuTotalStop - gpuTotalStart).count() / REPEATS;

    // PHASE 8: VERIFY RESULTS AND CLEAN UP

    float maxDifference = 0.0f;

    for (size_t i = 0; i < cpuOutput.size(); ++i) {
        float difference = std::fabs(cpuOutput[i] - gpuOutput[i]);

        if (difference > maxDifference) {
            maxDifference = difference;
        }
    }

   
    // Calculate speedups
    double kernelSpeedup = cpuMs / gpuKernelMs;
    double endToEndSpeedup = cpuMs / gpuEndToEndMs;

    std::cout << "\nPerformance Results:\n";
    std::cout << std::fixed << std::setprecision(9);

    std::cout << "Average CPU compute time: "
              << cpuMs << " ms\n";

    std::cout << "Average GPU kernel-only time: "
              << gpuKernelMs << " ms\n";

    std::cout << "Average GPU end-to-end time: "
              << gpuEndToEndMs << " ms\n";

    std::cout << "Kernel-only speedup: "
              << kernelSpeedup << "x\n";

    std::cout << "End-to-end speedup: "
              << endToEndSpeedup << "x\n";

    std::cout << "Maximum absolute output difference: "
              << maxDifference << "\n";

    std::cout << "CPU and GPU outputs: "
              << (maxDifference < 1e-4f
                  ? "MATCH (within tolerance)"
                  : "DO NOT MATCH") << "\n";

    std::cout << "First 10 output values (GPU):\n";

    for (int i = 0; i < 10; ++i) {
        std::cout << gpuOutput[i] << " ";
    }
    std::cout << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cudaFree(d_X);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_z1);
    cudaFree(d_z2);

    std::cout << "\nExperiment 5 completed.\n";

    return (maxDifference < 1e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
