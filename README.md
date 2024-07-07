# Lua-Powered Neural Network from Scratch

This project is a basic implementation of a neural network in Lua without using any external libraries. The script demonstrates Lua's capability to handle complex algorithms. The neural network includes functionalities for creating a network architecture, training it with a simple dataset (like XOR or a basic classification problem), and predicting outcomes.

## Key Features

1. **Network Setup:** The script defines a simple network architecture, which is a feedforward neural network with configurable layers and neurons.
2. **Forward Propagation:** The mechanism for data to move through the network is implemented, including weighted inputs and activation functions (sigmoid).
3. **Backpropagation:** The training algorithm is coded to adjust the weights based on error rates using gradient descent.
4. **Training and Testing:** Functions to train the network with sample data and then validate it using a separate test set are integrated.
5. **User Interaction:** A simple interface is provided for users to input new data for predictions and to tweak network parameters.

## Getting Started

To get started with this project, follow these steps:

1. Make sure you have Lua installed on your system.
2. Clone or download this repository to your local machine.
3. Open the `neural-network.lua` file in a text editor.
4. Modify the network parameters, such as the number of input nodes, hidden nodes, and output nodes, according to your requirements.
5. Run the script using the Lua interpreter:

   ```
   lua neural-network.lua
   ```

6. The script will train the neural network using the XOR dataset and display the predictions for each input combination.

## Customization

You can customize the neural network by modifying the following parameters in the script:

- `input_nodes`: The number of input nodes in the network.
- `hidden_nodes`: The number of hidden nodes in the network.
- `output_nodes`: The number of output nodes in the network.
- `learning_rate`: The learning rate used during training.
- `training_data`: The dataset used for training the network.

Feel free to experiment with different network architectures, activation functions, and datasets to explore the capabilities of the neural network.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgements

This project was inspired by the desire to understand the fundamental concepts of neural networks and implement them from scratch using Lua. It serves as an educational resource for anyone interested in learning about neural networks and their implementation.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
