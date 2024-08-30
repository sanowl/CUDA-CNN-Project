#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Convolution kernel
__global__ void convolution2D(float* input, float* kernel, float* output, int input_height, int input_width, int kernel_size, int output_height, int output_width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width)
    {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                int input_row = row + i;
                int input_col = col + j;
                sum += input[input_row * input_width + input_col] * kernel[i * kernel_size + j];
            }
        }
        output[row * output_width + col] = sum;
    }
}

// Max pooling kernel
__global__ void maxPooling2D(float* input, float* output, int input_height, int input_width, int pool_size, int output_height, int output_width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width)
    {
        float max_val = -INFINITY;
        for (int i = 0; i < pool_size; ++i)
        {
            for (int j = 0; j < pool_size; ++j)
            {
                int input_row = row * pool_size + i;
                int input_col = col * pool_size + j;
                float val = input[input_row * input_width + input_col];
                max_val = fmaxf(max_val, val);
            }
        }
        output[row * output_width + col] = max_val;
    }
}

// ReLU activation kernel
__global__ void relu(float* input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        input[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Function to check CUDA errors
void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Wrapper functions for launching kernels
void launchConvolution2D(float* d_input, float* d_kernel, float* d_output, int input_height, int input_width, int kernel_size, int output_height, int output_width)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    convolution2D<<<gridDim, blockDim>>>(d_input, d_kernel, d_output, input_height, input_width, kernel_size, output_height, output_width);
    checkCudaErrors(cudaGetLastError());
}

void launchMaxPooling2D(float* d_input, float* d_output, int input_height, int input_width, int pool_size, int output_height, int output_width)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    maxPooling2D<<<gridDim, blockDim>>>(d_input, d_output, input_height, input_width, pool_size, output_height, output_width);
    checkCudaErrors(cudaGetLastError());
}

void launchRelu(float* d_input, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu<<<numBlocks, blockSize>>>(d_input, size);
    checkCudaErrors(cudaGetLastError());
}