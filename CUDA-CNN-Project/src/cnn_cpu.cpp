#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>

// Convolution forward pass
void convolution2D(const std::vector<float>& input, const std::vector<float>& kernel, std::vector<float>& output,
                   int input_height, int input_width, int kernel_size, int output_height, int output_width) {
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    sum += input[(i + k) * input_width + (j + l)] * kernel[k * kernel_size + l];
                }
            }
            output[i * output_width + j] = sum;
        }
    }
}

// ReLU activation
void relu(std::vector<float>& data) {
    for (auto& d : data) {
        d = std::max(0.0f, d);
    }
}

// Max pooling forward pass
void maxPooling2D(const std::vector<float>& input, std::vector<float>& output,
                  int input_height, int input_width, int pool_size, int output_height, int output_width) {
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float max_val = -std::numeric_limits<float>::max();
            for (int k = 0; k < pool_size; ++k) {
                for (int l = 0; l < pool_size; ++l) {
                    max_val = std::max(max_val, input[(i * pool_size + k) * input_width + (j * pool_size + l)]);
                }
            }
            output[i * output_width + j] = max_val;
        }
    }
}

// Fully connected layer forward pass
void fullyConnectedForward(const std::vector<float>& input, const std::vector<float>& weights, const std::vector<float>& bias,
                           std::vector<float>& output, int batch_size, int input_size, int output_size) {
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = bias[j];
            for (int k = 0; k < input_size; ++k) {
                sum += input[i * input_size + k] * weights[k * output_size + j];
            }
            output[i * output_size + j] = sum;
        }
    }
}

// Softmax function
void softmax(const std::vector<float>& input, std::vector<float>& output, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; ++i) {
        float max_val = *std::max_element(input.begin() + i * num_classes, input.begin() + (i + 1) * num_classes);
        float sum = 0.0f;
        for (int j = 0; j < num_classes; ++j) {
            float exp_val = std::exp(input[i * num_classes + j] - max_val);
            output[i * num_classes + j] = exp_val;
            sum += exp_val;
        }
        for (int j = 0; j < num_classes; ++j) {
            output[i * num_classes + j] /= sum;
        }
    }
}

// Cross-entropy loss function
float crossEntropyLoss(const std::vector<float>& predictions, const std::vector<int>& labels, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        int label = labels[i];
        float pred = std::max(predictions[i * num_classes + label], 1e-7f);  // Avoid log(0)
        loss -= std::log(pred);
    }
    return loss / batch_size;
}

// Backpropagation for fully connected layer
void backpropFC(const std::vector<float>& input, const std::vector<float>& grad_output,
                std::vector<float>& weights, std::vector<float>& bias,
                std::vector<float>& grad_input, int batch_size, int input_size, int output_size, float learning_rate) {
    // Initialize grad_input to zeros
    std::fill(grad_input.begin(), grad_input.end(), 0.0f);

    // Compute gradients
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float grad = grad_output[i * output_size + j];
            for (int k = 0; k < input_size; ++k) {
                grad_input[i * input_size + k] += grad * weights[k * output_size + j];
                weights[k * output_size + j] -= learning_rate * grad * input[i * input_size + k];
            }
            bias[j] -= learning_rate * grad;
        }
    }
}

// ReLU backward pass
void reluBackward(const std::vector<float>& input, std::vector<float>& grad_output) {
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] <= 0) {
            grad_output[i] = 0;
        }
    }
}

// Max pooling backward pass
void maxPoolingBackward(const std::vector<float>& input, const std::vector<float>& output, 
                        std::vector<float>& grad_output, std::vector<float>& grad_input,
                        int input_height, int input_width, int pool_size, int output_height, int output_width) {
    std::fill(grad_input.begin(), grad_input.end(), 0.0f);
    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            int max_idx = 0;
            float max_val = -std::numeric_limits<float>::max();
            for (int k = 0; k < pool_size; ++k) {
                for (int l = 0; l < pool_size; ++l) {
                    int idx = (i * pool_size + k) * input_width + (j * pool_size + l);
                    if (input[idx] > max_val) {
                        max_val = input[idx];
                        max_idx = idx;
                    }
                }
            }
            grad_input[max_idx] += grad_output[i * output_width + j];
        }
    }
}

// Convolution backward pass
void convolutionBackward(const std::vector<float>& input, const std::vector<float>& grad_output,
                         std::vector<float>& grad_input, std::vector<float>& grad_kernel,
                         const std::vector<float>& kernel, int input_height, int input_width,
                         int kernel_size, int output_height, int output_width, float learning_rate) {
    std::fill(grad_input.begin(), grad_input.end(), 0.0f);
    std::fill(grad_kernel.begin(), grad_kernel.end(), 0.0f);

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float grad = grad_output[i * output_width + j];
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    grad_input[(i + k) * input_width + (j + l)] += grad * kernel[k * kernel_size + l];
                    grad_kernel[k * kernel_size + l] += grad * input[(i + k) * input_width + (j + l)];
                }
            }
        }
    }

    // Update kernel weights
    for (size_t i = 0; i < kernel.size(); ++i) {
        grad_kernel[i] = kernel[i] - learning_rate * grad_kernel[i];
    }
}

// Function to initialize weights with random values
void initializeWeights(std::vector<float>& weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 0.1);
    for (auto& w : weights) {
        w = d(gen);
    }
}