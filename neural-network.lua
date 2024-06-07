-- Define the sigmoid activation function
local function sigmoid(x)
    return 1 / (1 + math.exp(-x))
end

-- Define the derivative of the sigmoid function
local function sigmoid_derivative(x)
    return x * (1 - x)
end

-- Define the neural network class
local NeuralNetwork = {}

-- Initialize the neural network
function NeuralNetwork:new(input_nodes, hidden_nodes, output_nodes)
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    -- Initialize weights randomly between -1 and 1
    self.weights_ih = {}
    for i = 1, self.input_nodes do
        self.weights_ih[i] = {}
        for j = 1, self.hidden_nodes do
            self.weights_ih[i][j] = math.random() * 2 - 1
        end
    end

    self.weights_ho = {}
    for i = 1, self.hidden_nodes do
        self.weights_ho[i] = {}
        for j = 1, self.output_nodes do
            self.weights_ho[i][j] = math.random() * 2 - 1
        end
    end

    -- Initialize biases randomly between -1 and 1
    self.bias_h = {}
    for i = 1, self.hidden_nodes do
        self.bias_h[i] = math.random() * 2 - 1
    end

    self.bias_o = {}
    for i = 1, self.output_nodes do
        self.bias_o[i] = math.random() * 2 - 1
    end
end

-- Train the neural network
function NeuralNetwork:train(inputs, targets, learning_rate)
    -- Forward propagation
    local hidden_inputs = {}
    for i = 1, self.hidden_nodes do
        hidden_inputs[i] = 0
        for j = 1, self.input_nodes do
            hidden_inputs[i] = hidden_inputs[i] + inputs[j] * self.weights_ih[j][i]
        end
        hidden_inputs[i] = hidden_inputs[i] + self.bias_h[i]
    end

    local hidden_outputs = {}
    for i = 1, self.hidden_nodes do
        hidden_outputs[i] = sigmoid(hidden_inputs[i])
    end

    local final_inputs = {}
    for i = 1, self.output_nodes do
        final_inputs[i] = 0
        for j = 1, self.hidden_nodes do
            final_inputs[i] = final_inputs[i] + hidden_outputs[j] * self.weights_ho[j][i]
        end
        final_inputs[i] = final_inputs[i] + self.bias_o[i]
    end

    local final_outputs = {}
    for i = 1, self.output_nodes do
        final_outputs[i] = sigmoid(final_inputs[i])
    end

    -- Backpropagation
    local output_errors = {}
    for i = 1, self.output_nodes do
        output_errors[i] = targets[i] - final_outputs[i]
    end

    local hidden_errors = {}
    for i = 1, self.hidden_nodes do
        hidden_errors[i] = 0
        for j = 1, self.output_nodes do
            hidden_errors[i] = hidden_errors[i] + output_errors[j] * self.weights_ho[i][j]
        end
    end

    -- Update weights and biases
    for i = 1, self.hidden_nodes do
        for j = 1, self.output_nodes do
            self.weights_ho[i][j] = self.weights_ho[i][j] + learning_rate * output_errors[j] * sigmoid_derivative(final_outputs[j]) * hidden_outputs[i]
        end
    end

    for i = 1, self.input_nodes do
        for j = 1, self.hidden_nodes do
            self.weights_ih[i][j] = self.weights_ih[i][j] + learning_rate * hidden_errors[j] * sigmoid_derivative(hidden_outputs[j]) * inputs[i]
        end
    end

    for i = 1, self.hidden_nodes do
        self.bias_h[i] = self.bias_h[i] + learning_rate * hidden_errors[i] * sigmoid_derivative(hidden_outputs[i])
    end

    for i = 1, self.output_nodes do
        self.bias_o[i] = self.bias_o[i] + learning_rate * output_errors[i] * sigmoid_derivative(final_outputs[i])
    end
end

-- Predict the output for given inputs
function NeuralNetwork:predict(inputs)
    local hidden_inputs = {}
    for i = 1, self.hidden_nodes do
        hidden_inputs[i] = 0
        for j = 1, self.input_nodes do
            hidden_inputs[i] = hidden_inputs[i] + inputs[j] * self.weights_ih[j][i]
        end
        hidden_inputs[i] = hidden_inputs[i] + self.bias_h[i]
    end

    local hidden_outputs = {}
    for i = 1, self.hidden_nodes do
        hidden_outputs[i] = sigmoid(hidden_inputs[i])
    end

    local final_inputs = {}
    for i = 1, self.output_nodes do
        final_inputs[i] = 0
        for j = 1, self.hidden_nodes do
            final_inputs[i] = final_inputs[i] + hidden_outputs[j] * self.weights_ho[j][i]
        end
        final_inputs[i] = final_inputs[i] + self.bias_o[i]
    end

    local final_outputs = {}
    for i = 1, self.output_nodes do
        final_outputs[i] = sigmoid(final_inputs[i])
    end

    return final_outputs
end

-- Example usage
local nn = NeuralNetwork:new(2, 4, 1)

-- XOR dataset
local training_data = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 0}
}

-- Training loop
for i = 1, 10000 do
    local index = math.random(1, #training_data)
    local inputs = {training_data[index][1], training_data[index][2]}
    local targets = {training_data[index][3]}
    nn:train(inputs, targets, 0.1)
end

-- Testing
print("0 XOR 0 = " .. nn:predict({0, 0})[1])
print("0 XOR 1 = " .. nn:predict({0, 1})[1])
print("1 XOR 0 = " .. nn:predict({1, 0})[1])
print("1 XOR 1 = " .. nn:predict({1, 1})[1])
