import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 16,
    ):
        """Initialize a simple feedforward neural network.

        Network structure: input -> hidden -> output
        Activation: tanh on both layers

        Args:
            input_size: Dimension of input (observation size)
            output_size: Dimension of output (action size)
            hidden_size: Number of hidden neurons
        """
        # Here we randomly initialize our neural network layers,
        # as well as our input and output size.
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        # TODO: Initialize weight matrices with uniform random values in [-1, 1]
        # - self.input_to_hidden: shape (hidden_size, input_size)
        # - self.hidden_to_output: shape (output_size, hidden_size)
        # Hint: Use np.random.uniform(-1, 1, (rows, cols))

        self.input_to_hidden = np.random.uniform(low=-1.0, high=1.0, size=(self.n_hidden, self.n_input))
        self.hidden_to_output = np.random.uniform(low=-1.0, high=1.0, size=(self.n_output, self.n_hidden))

        # self.bias_hidden = np.zeros(self.n_hidden)
        # self.bias_output = np.zeros(self.n_output)

        # TODO: Compute number of parameters in each layer
        # - self.n_params_i2h = input_size * hidden_size
        # - self.n_params_h2o = hidden_size * output_size
        self.n_params_i2h = self.n_hidden * self.n_input  
        self.n_params_h2o = self.n_output * self.n_hidden 

        # Nombre de biais
        self.n_bias_hidden = 0 # self.n_hidden 
        self.n_bias_output = 0 # self.n_output

        self.n_params = self.get_num_params()


    def get_action(self, state):
        """Forward pass through the network.

        Args:
            state: Observation array, shape (input_size,) or (batch_size, input_size)

        Returns:
            action: Output array, shape (output_size,) or (batch_size, output_size)
        """
        # TODO: Perform forward pass computation
        # 1. Hidden layer: hidden = tanh(state @ input_to_hidden.T)
        # 2. Output layer: output = tanh(hidden @ hidden_to_output.T)
        # 3. Clip output to [-1, 1] using np.clip()
        #
        # Hint: Use @ operator or np.matmul for matrix multiplication
        # Hint: .T transposes a matrix
        # Hint: np.tanh() applies tanh element-wise
        hidden = np.tanh(state @ self.input_to_hidden.T ) # + self.bias_hidden
        output = np.tanh(hidden @ self.hidden_to_output.T ) # + self.bias_output
        action = np.clip(output, -1.0, 1.0)
        return action 

    def set_weights(self, encoding):
        """Set network weights from a flat parameter vector.

        Args:
            encoding: Flat array of size (n_params,) containing all weights
        """
        # TODO: Map the flat encoding to weight matrices
        # 1. Split encoding into two parts:
        #    - First n_params_i2h values for input_to_hidden
        #    - Remaining n_params_h2o values for hidden_to_output
        # 2. Reshape each part to match the weight matrix shapes
        #
        # Hint: Use array slicing: encoding[:n] and encoding[n:]
        # Hint: Use np.reshape(array, (rows, cols)) or array.reshape((rows, cols))
        assert len(encoding) == self.n_params, f"Expected {self.n_params} parameters, got {len(encoding)}"
        
        
        i2h_weights = encoding[:self.n_params_i2h].reshape((self.n_hidden, self.n_input))
        h2o_weights = encoding[self.n_params_i2h:].reshape((self.n_output, self.n_hidden))
        self.input_to_hidden = i2h_weights
        self.hidden_to_output = h2o_weights 
        """
        idx = 0
        self.input_to_hidden = encoding[idx : idx + self.n_params_i2h].reshape((self.n_hidden, self.n_input))
        idx += self.n_params_i2h
        
        self.hidden_to_output = encoding[idx : idx + self.n_params_h2o].reshape((self.n_output, self.n_hidden))
        idx += self.n_params_h2o
        
        self.bias_hidden = encoding[idx : idx + self.n_bias_hidden]
        idx += self.n_bias_hidden
        
        self.bias_output = encoding[idx : idx + self.n_bias_output]
        """
        return

    def geno2pheno(self, genotype):
        """Alias for set_weights (genotype to phenotype mapping)."""
        self.set_weights(genotype)

    def get_num_params(self):
        # To provide a genetic encoding for our neural network controller,
        # we compute and store the number of parameters in our NN class.
        # TODO: Return sum of parameters in both layers!
        return self.n_params_i2h + self.n_params_h2o # + self.n_bias_hidden + self.n_bias_output


    def reset_controller(self, batch_size=1) -> None:
        pass
