from torch import nn
import torch
import math
import numpy as np

from .spice_som import SpiceSOM

class SpiceHCM(nn.Module):
    def __init__(self,
                 som_1_length: int,
                 som_2_length: int,
                 const_lrf_weights: float,
                 const_lrf_trust_of_new: float):
        """
        This object represents a hebbian correlation matrix between 2 SOMs.
        :param som_1: The first SOM.
        :param som_2: The second SOM.
        :param lrf_weights: The learning rate of the weights, how strong each update affects the matrix.
        :param lrf_trust_of_new: This implementation uses the hebbian covariance learning mechanism. This uses an 'average value', this value is update continuous in this implementation. Use this parameter to tell how much you trust that the new value and want it to change the average. For more information on the mechanism look here: https://rkypragada.medium.com/hebbian-learning-c2166ac0f48d
        """
        super().__init__()
        self.__lrf_trust_of_new = const_lrf_trust_of_new
        self.__lrf_weights = const_lrf_weights
        self.weights = np.ones((som_1_length, som_2_length))
        self.activation_bar_vector_1: np.array = np.zeros(som_1_length)
        self.activation_bar_vector_2: np.array = np.zeros(som_2_length)
        self.__som_1_length = som_1_length
        self.__som_2_length = som_2_length
        self.__iteration = 0
        
    @classmethod
    def from_weights(cls, weights: np.ndarray, activation_bar_vector_1: np.ndarray, activation_bar_vector_2: np.ndarray, const_lrf_weights: float, const_lrf_trust_of_new: float) -> "SpiceNetHcm":
        hcm = cls(weights.shape[0], weights.shape[1], const_lrf_weights, const_lrf_trust_of_new)
        hcm.weights = weights
        hcm.activation_bar_vector_1 = activation_bar_vector_1
        hcm.activation_bar_vector_2 = activation_bar_vector_2
        return hcm
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        results = []
        for i in range(len(input)):
            activation_vector_som_1 = input[i, 0].numpy()
            activation_vector_som_2 = input[i, 1].numpy()
            
            # We dont want to change any parameter during the forward pass, so we dont automatically update the activation bar vector.
            loc_activation_bar_vector_1 = ((1.0 - self.__lrf_trust_of_new)
                                                * self.activation_bar_vector_1
                                                + self.__lrf_trust_of_new
                                                * activation_vector_som_1)
            loc_activation_bar_vector_2 = ((1.0 - self.__lrf_trust_of_new)
                                                * self.activation_bar_vector_2
                                                + self.__lrf_trust_of_new
                                                * activation_vector_som_2)

            # Som 1 represents the y-axis and Som 2 the x-axis
            weights_delta_matrix = (
                    self.__lrf_weights
                    * np.matrix(activation_vector_som_1 - loc_activation_bar_vector_1)
                    .transpose()
                    .dot(np.matrix(activation_vector_som_2 - loc_activation_bar_vector_2))
            )
            results.append((self.weights + weights_delta_matrix))
            
        # We only return the matrix, maybe should return new activation bar vectors too?
        return torch.from_numpy(np.array(results))
    
    def get_soms(self):
        return self.__som_1, self.__som_2

    def get_matrix(self) -> np.ndarray:
        """
        Get the weight matrix.
        :return: The matrix y-axis is the first som given to the constructor (x-axis the second one).
        """
        return self.weights
    
    def calculate_som_1_to_2(self, a_array: np.array) -> np.array:
        if a_array.shape != (self.__som_1_length,):
            raise ValueError("The input vector must be of the same length as the som amount of neurons.")
        # Take each column of the weight matrix and calculate the dot product with the input array.
        return np.array([a_array.dot(self.weights[:, i]) for i in range(self.weights.shape[0])])

    def calculate_som_2_to_1(self, a_array: np.array) -> np.array:
        if a_array.shape != (self.__som_2_length,):
            raise ValueError("The input vector must be of the same length as the som amount of neurons.")
        # Take each column of the weight matrix and calculate the dot product with the input array.
        return np.array([a_array.dot(self.weights[i, :]) for i in range(self.weights.shape[1])])
    
    def fit(self,
            som_1,
            som_2,
            values_som_1: list[float],
            values_som_2: list[float],
            epochs: int):
        """
        Fits the hebbian correlation matrix.

        :param values_som_1: The values of the first som.
        :param values_som_2: The values of the second som.
        :param epochs: How many times the matrix is fitted to the data.
        """
        if len(values_som_1) != len(values_som_2):
            raise Exception('The length of the values of the first som must be equal to the length of the second som.')
        
        if len(som_1) != self.__som_1_length:
            raise Exception('The first SOM must have the same amount of neurons as the length of the first dimension of the weight matrix.')
        if len(som_2) != self.__som_2_length:
            raise Exception('The second SOM must have the same amount of neurons as the length of the second dimension of the weight matrix.')
        
        if len(som_1) != len(som_2):
            raise Exception('The first and second SOM must have the same amount of neurons.')

        for _ in range(epochs):
            for i in range(len(values_som_1)):
                activation_vector_som_1 = som_1.get_activation_vector(values_som_1[i])
                activation_vector_som_2 = som_2.get_activation_vector(values_som_2[i])
                self.activation_bar_vector_1 = ((1.0 - self.__lrf_trust_of_new)
                                                  * self.activation_bar_vector_1
                                                  + self.__lrf_trust_of_new
                                                  * activation_vector_som_1)
                self.activation_bar_vector_2 = ((1.0 - self.__lrf_trust_of_new)
                                                  * self.activation_bar_vector_2
                                                  + self.__lrf_trust_of_new
                                                  * activation_vector_som_2)

                # Som 1 represents the y-axis and Som 2 the x-axis
                weights_delta_matrix = (
                        self.__lrf_weights
                        * np.matrix(activation_vector_som_1 - self.activation_bar_vector_1)
                        .transpose()
                        .dot(np.matrix(activation_vector_som_2 - self.activation_bar_vector_2))
                )
                self.weights += weights_delta_matrix
                self.__iteration += 1
