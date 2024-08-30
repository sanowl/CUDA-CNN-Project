#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cmath>
#include "cnn.cu"

// Function to load data from file
void loadData(const char* filename, float* data, int size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    fread(data, sizeof(float), size, file);
    fclose(file);
}

// Function to initialize weights with random values
__global__ void initializeWeights(float* weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state) * 0.1f;
    }
}

// Softmax kernel
__global__ void softmax(float* input, float* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float max_val = -INFINITY;
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[idx * num_classes + i]);
        }
        
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[idx * num_classes + i] - max_val);
            output[idx * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        for (int i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] /= sum;
        }
    }
}

// Cross-entropy loss kernel
__global__ void crossEntropyLoss(float* predictions, int* labels, float* loss, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int label = labels[idx];
        float pred = fmaxf(predictions[idx * num_classes + label], 1e-7f);  // Avoid log(0)
        atomicAdd(loss, -logf(pred));
    }
}

// Backpropagation kernel for the fully connected layer
__global__ void backpropFC(float* d_fc, float* d_fc_error, float* d_pool2, float* d_fc_weights, float* d_fc_bias, 
                           int batch_size, int input_size, int output_size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        for (int i = 0; i < batch_size; i++) {
            float error = d_fc_error[i * output_size + idx];
            for (int j = 0; j < input_size; j++) {
                float gradient = error * d_pool2[i * input_size + j];
                atomicAdd(&d_fc_weights[j * output_size + idx], -learning_rate * gradient);
            }
            atomicAdd(&d_fc_bias[idx], -learning_rate * error);
        }
    }
}

int main() {
    // Hyperparameters
    const int input_size = 28 * 28;
    const int num_classes = 10;
    const int batch_size = 64;
    const int num_epochs = 10;
    const float learning_rate = 0.01f;

    // Load data
    float* h_input = (float*)malloc(batch_size * input_size * sizeof(float));
    int* h_labels = (int*)malloc(batch_size * sizeof(int));
    loadData("data/X_train.npy", h_input, batch_size * input_size);
    loadData("data/y_train.npy", (float*)h_labels, batch_size);

    // Allocate device memory
    float *d_input, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc, *d_output;
    int *d_labels;
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_conv1, batch_size * 26 * 26 * sizeof(float));
    cudaMalloc(&d_pool1, batch_size * 13 * 13 * sizeof(float));
    cudaMalloc(&d_conv2, batch_size * 11 * 11 * sizeof(float));
    cudaMalloc(&d_pool2, batch_size * 5 * 5 * sizeof(float));
    cudaMalloc(&d_fc, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_classes * sizeof(float));
    cudaMalloc(&d_labels, batch_size * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // Define CNN architecture
    const int conv1_size = 3;
    const int conv2_size = 3;
    const int pool_size = 2;

    // Allocate and initialize kernels and weights
    float *d_conv1_kernel, *d_conv2_kernel, *d_fc_weights, *d_fc_bias;
    cudaMalloc(&d_conv1_kernel, conv1_size * conv1_size * sizeof(float));
    cudaMalloc(&d_conv2_kernel, conv2_size * conv2_size * sizeof(float));
    cudaMalloc(&d_fc_weights, 5 * 5 * num_classes * sizeof(float));
    cudaMalloc(&d_fc_bias, num_classes * sizeof(float));

    dim3 initBlockSize(256);
    dim3 initGridSize((conv1_size * conv1_size + initBlockSize.x - 1) / initBlockSize.x);
    initializeWeights<<<initGridSize, initBlockSize>>>(d_conv1_kernel, conv1_size * conv1_size, time(NULL));
    initGridSize = dim3((conv2_size * conv2_size + initBlockSize.x - 1) / initBlockSize.x);
    initializeWeights<<<initGridSize, initBlockSize>>>(d_conv2_kernel, conv2_size * conv2_size, time(NULL));
    initGridSize = dim3((5 * 5 * num_classes + initBlockSize.x - 1) / initBlockSize.x);
    initializeWeights<<<initGridSize, initBlockSize>>>(d_fc_weights, 5 * 5 * num_classes, time(NULL));
    initGridSize = dim3((num_classes + initBlockSize.x - 1) / initBlockSize.x);
    initializeWeights<<<initGridSize, initBlockSize>>>(d_fc_bias, num_classes, time(NULL));

    // Allocate memory for gradients
    float *d_conv1_grad, *d_conv2_grad, *d_fc_grad;
    cudaMalloc(&d_conv1_grad, conv1_size * conv1_size * sizeof(float));
    cudaMalloc(&d_conv2_grad, conv2_size * conv2_size * sizeof(float));
    cudaMalloc(&d_fc_grad, batch_size * num_classes * sizeof(float));

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;

        // Forward pass
        launchConvolution2D(d_input, d_conv1_kernel, d_conv1, 28, 28, conv1_size, 26, 26);
        launchRelu(d_conv1, batch_size * 26 * 26);
        launchMaxPooling2D(d_conv1, d_pool1, 26, 26, pool_size, 13, 13);

        launchConvolution2D(d_pool1, d_conv2_kernel, d_conv2, 13, 13, conv2_size, 11, 11);
        launchRelu(d_conv2, batch_size * 11 * 11);
        launchMaxPooling2D(d_conv2, d_pool2, 11, 11, pool_size, 5, 5);

        // Fully connected layer
        dim3 fcBlockSize(256);
        dim3 fcGridSize((batch_size * num_classes + fcBlockSize.x - 1) / fcBlockSize.x);
        matrixMultiply<<<fcGridSize, fcBlockSize>>>(d_pool2, d_fc_weights, d_fc_bias, d_fc, batch_size, 5 * 5, num_classes);

        // Softmax
        dim3 softmaxBlockSize(256);
        dim3 softmaxGridSize((batch_size + softmaxBlockSize.x - 1) / softmaxBlockSize.x);
        softmax<<<softmaxGridSize, softmaxBlockSize>>>(d_fc, d_output, batch_size, num_classes);

        // Calculate loss
        float* d_loss;
        cudaMalloc(&d_loss, sizeof(float));
        cudaMemset(d_loss, 0, sizeof(float));
        dim3 lossBlockSize(256);
        dim3 lossGridSize((batch_size + lossBlockSize.x - 1) / lossBlockSize.x);
        crossEntropyLoss<<<lossGridSize, lossBlockSize>>>(d_output, d_labels, d_loss, batch_size, num_classes);
        cudaMemcpy(&total_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
        total_loss /= batch_size;

        // Backpropagation
        // ... (implement backpropagation for each layer)

        // Update weights
        backpropFC<<<fcGridSize, fcBlockSize>>>(d_fc, d_fc_grad, d_pool2, d_fc_weights, d_fc_bias, 
                                                batch_size, 5 * 5, num_classes, learning_rate);

        // Print epoch results
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss);

        cudaFree(d_loss);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_conv1);
    cudaFree(d_pool1);
    cudaFree(d_conv2);
    cudaFree(d_pool2);
    cudaFree(d_fc);
    cudaFree(d_output);
    cudaFree(d_labels);
    cudaFree(d_conv1_kernel);
    cudaFree(d_conv2_kernel);
    cudaFree(d_fc_weights);
    cudaFree(d_fc_bias);
    cudaFree(d_conv1_grad);
    cudaFree(d_conv2_grad);
    cudaFree(d_fc_grad);
    free(h_input);
    free(h_labels);

    return 0;
}