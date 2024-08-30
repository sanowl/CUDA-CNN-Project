#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <curand_kernel.h>

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
__global__ void maxPooling2D(float* input, float* output, int* indices, int input_height, int input_width, int pool_size, int output_height, int output_width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width)
    {
        float max_val = -INFINITY;
        int max_idx = 0;
        for (int i = 0; i < pool_size; ++i)
        {
            for (int j = 0; j < pool_size; ++j)
            {
                int input_row = row * pool_size + i;
                int input_col = col * pool_size + j;
                int idx = input_row * input_width + input_col;
                float val = input[idx];
                if (val > max_val)
                {
                    max_val = val;
                    max_idx = idx;
                }
            }
        }
        output[row * output_width + col] = max_val;
        indices[row * output_width + col] = max_idx;
    }
}

// ReLU activation kernel
__global__ void relu(float* input, float* output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Softmax kernel
__global__ void softmax(float* input, float* output, int batch_size, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; ++i)
        {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i)
        {
            float exp_val = expf(input[idx * num_classes + i] - max_val);
            output[idx * num_classes + i] = exp_val;
            sum += exp_val;
        }

        for (int i = 0; i < num_classes; ++i)
        {
            output[idx * num_classes + i] /= sum;
        }
    }
}

// Cross-entropy loss kernel
__global__ void crossEntropyLoss(float* predictions, int* labels, float* loss, int batch_size, int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size)
    {
        int label = labels[idx];
        float pred = fmaxf(predictions[idx * num_classes + label], 1e-7f);  // Avoid log(0)
        atomicAdd(loss, -logf(pred));
    }
}

// Convolution backward kernel
__global__ void convolutionBackward(float* input, float* grad_output, float* grad_input, float* grad_kernel,
                                    float* kernel, int input_height, int input_width, int kernel_size,
                                    int output_height, int output_width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width)
    {
        float grad = grad_output[row * output_width + col];
        for (int i = 0; i < kernel_size; ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                int input_row = row + i;
                int input_col = col + j;
                atomicAdd(&grad_input[input_row * input_width + input_col], grad * kernel[i * kernel_size + j]);
                atomicAdd(&grad_kernel[i * kernel_size + j], grad * input[input_row * input_width + input_col]);
            }
        }
    }
}

// Max pooling backward kernel
__global__ void maxPoolingBackward(float* grad_output, int* indices, float* grad_input,
                                   int input_height, int input_width, int output_height, int output_width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width)
    {
        int idx = row * output_width + col;
        int max_idx = indices[idx];
        atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
}

// ReLU backward kernel
__global__ void reluBackward(float* input, float* grad_output, float* grad_input, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad_input[idx] = (input[idx] > 0) ? grad_output[idx] : 0;
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

// Weight initialization kernel
__global__ void initializeWeights(float* weights, int size, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state) * 0.1f;
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

void launchMaxPooling2D(float* d_input, float* d_output, int* d_indices, int input_height, int input_width, int pool_size, int output_height, int output_width)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    maxPooling2D<<<gridDim, blockDim>>>(d_input, d_output, d_indices, input_height, input_width, pool_size, output_height, output_width);
    checkCudaErrors(cudaGetLastError());
}

void launchRelu(float* d_input, float* d_output, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    relu<<<numBlocks, blockSize>>>(d_input, d_output, size);
    checkCudaErrors(cudaGetLastError());
}

void launchSoftmax(float* d_input, float* d_output, int batch_size, int num_classes)
{
    int blockSize = 256;
    int numBlocks = (batch_size + blockSize - 1) / blockSize;
    softmax<<<numBlocks, blockSize>>>(d_input, d_output, batch_size, num_classes);
    checkCudaErrors(cudaGetLastError());
}

void launchCrossEntropyLoss(float* d_predictions, int* d_labels, float* d_loss, int batch_size, int num_classes)
{
    int blockSize = 256;
    int numBlocks = (batch_size + blockSize - 1) / blockSize;
    crossEntropyLoss<<<numBlocks, blockSize>>>(d_predictions, d_labels, d_loss, batch_size, num_classes);
    checkCudaErrors(cudaGetLastError());
}

void launchConvolutionBackward(float* d_input, float* d_grad_output, float* d_grad_input, float* d_grad_kernel,
                               float* d_kernel, int input_height, int input_width, int kernel_size,
                               int output_height, int output_width)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    convolutionBackward<<<gridDim, blockDim>>>(d_input, d_grad_output, d_grad_input, d_grad_kernel,
                                               d_kernel, input_height, input_width, kernel_size,
                                               output_height, output_width);
    checkCudaErrors(cudaGetLastError());
}

void launchMaxPoolingBackward(float* d_grad_output, int* d_indices, float* d_grad_input,
                              int input_height, int input_width, int output_height, int output_width)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    maxPoolingBackward<<<gridDim, blockDim>>>(d_grad_output, d_indices, d_grad_input,
                                              input_height, input_width, output_height, output_width);
    checkCudaErrors(cudaGetLastError());
}

void launchReluBackward(float* d_input, float* d_grad_output, float* d_grad_input, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    reluBackward<<<numBlocks, blockSize>>>(d_input, d_grad_output, d_grad_input, size);
    checkCudaErrors(cudaGetLastError());
}

void launchInitializeWeights(float* d_weights, int size, unsigned long seed)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    initializeWeights<<<numBlocks, blockSize>>>(d_weights, size, seed);
    checkCudaErrors(cudaGetLastError());
}