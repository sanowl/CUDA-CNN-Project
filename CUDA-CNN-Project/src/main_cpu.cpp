#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

// Function to load data from file
void loadData(const char* filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
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

int main() {
    // Hyperparameters
    const int input_size = 28 * 28;
    const int num_classes = 10;
    const int batch_size = 64;
    const int num_epochs = 10;
    const float learning_rate = 0.01f;

    // Load data
    std::vector<float> input(batch_size * input_size);
    std::vector<int> labels(batch_size);
    loadData("data/X_train.npy", input);
    loadData("data/y_train.npy", reinterpret_cast<std::vector<float>&>(labels));

    // Define CNN architecture
    const int conv1_size = 3;
    const int conv2_size = 3;
    const int pool_size = 2;

    // Initialize layers
    std::vector<float> conv1(batch_size * 26 * 26);
    std::vector<float> pool1(batch_size * 13 * 13);
    std::vector<float> conv2(batch_size * 11 * 11);
    std::vector<float> pool2(batch_size * 5 * 5);
    std::vector<float> fc(batch_size * num_classes);
    std::vector<float> output(batch_size * num_classes);

    // Initialize weights and biases
    std::vector<float> conv1_kernel(conv1_size * conv1_size);
    std::vector<float> conv2_kernel(conv2_size * conv2_size);
    std::vector<float> fc_weights(5 * 5 * num_classes);
    std::vector<float> fc_bias(num_classes);

    initializeWeights(conv1_kernel);
    initializeWeights(conv2_kernel);
    initializeWeights(fc_weights);
    initializeWeights(fc_bias);

    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;

        // Forward pass
        convolution2D(input, conv1_kernel, conv1, 28, 28, conv1_size, 26, 26);
        relu(conv1);
        maxPooling2D(conv1, pool1, 26, 26, pool_size, 13, 13);

        convolution2D(pool1, conv2_kernel, conv2, 13, 13, conv2_size, 11, 11);
        relu(conv2);
        maxPooling2D(conv2, pool2, 11, 11, pool_size, 5, 5);

        fullyConnectedForward(pool2, fc_weights, fc_bias, fc, batch_size, 5 * 5, num_classes);
        softmax(fc, output, batch_size, num_classes);

        // Calculate loss
        total_loss = crossEntropyLoss(output, labels, batch_size, num_classes);

        // Backpropagation
        std::vector<float> grad_output(batch_size * num_classes);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < num_classes; ++j) {
                grad_output[i * num_classes + j] = output[i * num_classes + j];
                if (j == labels[i]) {
                    grad_output[i * num_classes + j] -= 1.0f;
                }
            }
        }

        std::vector<float> grad_fc(batch_size * 5 * 5);
        backpropFC(pool2, grad_output, fc_weights, fc_bias, grad_fc, batch_size, 5 * 5, num_classes, learning_rate);

        std::vector<float> grad_pool2(batch_size * 5 * 5);
        maxPoolingBackward(conv2, pool2, grad_fc, grad_pool2, 11, 11, pool_size, 5, 5);

        std::vector<float> grad_conv2(batch_size * 11 * 11);
        reluBackward(conv2, grad_pool2);

        std::vector<float> grad_pool1(batch_size * 13 * 13);
        std::vector<float> grad_conv2_kernel(conv2_size * conv2_size);
        convolutionBackward(pool1, grad_pool2, grad_pool1, grad_conv2_kernel, conv2_kernel, 13, 13, conv2_size, 11, 11, learning_rate);

        std::vector<float> grad_conv1(batch_size * 26 * 26);
        maxPoolingBackward(conv1, pool1, grad_pool1, grad_conv1, 26, 26, pool_size, 13, 13);

        reluBackward(conv1, grad_conv1);

        std::vector<float> grad_input(batch_size * input_size);
        std::vector<float> grad_conv1_kernel(conv1_size * conv1_size);
        convolutionBackward(input, grad_conv1, grad_input, grad_conv1_kernel, conv1_kernel, 28, 28, conv1_size, 26, 26, learning_rate);

        // Update weights
        for (size_t i = 0; i < conv1_kernel.size(); ++i) {
            conv1_kernel[i] = grad_conv1_kernel[i];
        }
        for (size_t i = 0; i < conv2_kernel.size(); ++i) {
            conv2_kernel[i] = grad_conv2_kernel[i];
        }

        // Print epoch results
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss << std::endl;
    }

    return 0;
}