import time
from typing import Callable, Optional
from torch import nn
import torch
import math
import numpy as np
from tqdm import tqdm
import copy

from .spice_som import SpiceSOM
from .spice_hcm import SpiceHCM

class SpiceNet(nn.Module):
    def __init__(self,
                 spice_som_1: SpiceSOM,
                 spice_som_2: SpiceSOM,
                 correlation_matrix: SpiceHCM):
        super().__init__()
        self.som_1 = [spice_som_1]
        self.som_2 = [spice_som_2]
        self.__correlation_matrix = [correlation_matrix]
        
        # To allow for the forward step to be called in NIRTorch we need to the last som and correlation matrix to be stored
        # Maybe there can be a better way in the future
        self.last_som_1 = None
        self.last_som_2 = None
        self.last_correlation_matrix = None
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Store last som and correlation matrix
        # This is used during nir export to ensure that it gets the weights before the NIRTorch forward step is called
        # Preferably avoid this issue all together by setting the model into eval mode before exporting
        self.last_som_1 = copy.deepcopy(self.som_1)
        self.last_som_2 = copy.deepcopy(self.som_2)
        self.last_correlation_matrix = copy.deepcopy(self.__correlation_matrix)
        
        values_som_1 = input[:, 0].numpy()
        values_som_2 = input[:, 1].numpy()
        b_size = len(input)
        p_list_som_1 = [values_som_1[i:i + b_size] for i in range(0, len(values_som_1), b_size)]
        p_list_som_2 = [values_som_2[i:i + b_size] for i in range(0, len(values_som_2), b_size)]

        iterator = range(len(p_list_som_1))
        
        # Only fit if the model is in training mode, so in eval mode the weights are not updated
        if self.training:
            for i in iterator:
                self.som_1[0].fit(p_list_som_1[i], 10)
                self.som_2[0].fit(p_list_som_2[i], 10)

                self.__correlation_matrix[0].fit(som_1=self.som_1[0],
                                            som_2=self.som_2[0],
                                            values_som_1=p_list_som_1[i],
                                            values_som_2=p_list_som_2[i],
                                            epochs=10)
            
        # Again need to copy here to not return reference
        return torch.from_numpy(self.__correlation_matrix[0].get_matrix().copy())

    def get_som_1(self) -> SpiceSOM:
        return self.som_1[0]

    def get_som_2(self) -> SpiceSOM:
        return self.som_2[0]

    def get_correlation_matrix(self) -> SpiceHCM:
        return self.__correlation_matrix[0]

    def decode(self, som_1_value: Optional[float] = None, som_2_value: Optional[float] = None) -> float:
        if som_1_value is None and som_2_value is None:
            raise ValueError('som_1_value and som_2_value cannot be both None')

        if som_1_value is not None:
            activation_values = self.som_1[0].get_activation_vector(som_1_value)
            som_2_should_activations = self.__correlation_matrix[0].calculate_som_1_to_2(activation_values)
            winner_index = som_2_should_activations.argmax()

            return self.som_2[0].naive_decode(som_2_should_activations[winner_index], winner_index)
        else:
            activation_values = self.som_2[0].get_activation_vector(som_2_value)
            som_1_should_activations = self.__correlation_matrix[0].calculate_som_2_to_1(activation_values)
            winner_index = som_1_should_activations.argmax()

            return self.som_1[0].naive_decode(som_1_should_activations[winner_index], winner_index)

    def fit(self,
            values_som_1: list[float],
            values_som_2: list[float],
            epochs_on_batch: int,
            batch_size: Optional[int] = None,
            after_batch_callback: Optional[Callable] = None,
            print_output: Optional[bool] = False):
        """
        Use this class to train and use a SpiceNet
        :param values_som_1: The values for the som no. 1.
        :param values_som_2: The values for the som no. 2.
        :param epochs_on_batch: Define how often a batch is used for training.
        :param batch_size: The values will be split in batches of this size for the training.
        :param after_batch_callback: This method will be called after each training with a batch. (Use it for plotting or what ever)
        :param print_output: This will toggle a progressbar implemented with tqdm and stops the computation times.
        :return:
        """
        som_elapsed_time = 0
        cm_elapsed_time = 0
        tmp: float = 0.0

        if len(values_som_1) != len(values_som_2):
            raise Exception('The length of the values of the first som must be equal to the length of the second som.')

        b_size = len(values_som_1) if batch_size is None else batch_size
        p_list_som_1 = [values_som_1[i:i + b_size] for i in range(0, len(values_som_1), b_size)]
        p_list_som_2 = [values_som_2[i:i + b_size] for i in range(0, len(values_som_2), b_size)]

        iterator = range(len(p_list_som_1)) if print_output is False else tqdm(range(len(p_list_som_1)), colour='green')
        for i in iterator:
            if print_output:
                tmp = time.time()
            self.som_1[0].fit(p_list_som_1[i], epochs_on_batch)
            self.som_2[0].fit(p_list_som_2[i], epochs_on_batch)
            if print_output:
                som_elapsed_time += time.time() - tmp

            if print_output:
                tmp = time.time()
            self.__correlation_matrix[0].fit(som_1=self.som_1[0],
                                          som_2=self.som_2[0],
                                          values_som_1=p_list_som_1[i],
                                          values_som_2=p_list_som_2[i],
                                          epochs=epochs_on_batch)
            if print_output:
                cm_elapsed_time += time.time() - tmp

            if after_batch_callback is not None:
                after_batch_callback()

        if print_output:
            print(f'Time spend on the Components: \nSom: {som_elapsed_time} s | Convolution Matrix: {cm_elapsed_time} s')
