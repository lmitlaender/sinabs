import time
from typing import Callable, Optional
from torch import nn
import torch
import math
import numpy as np
from tqdm import tqdm

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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

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
            self.__correlation_matrix[0].fit(som_1=self.som_1,
                                          som_2=self.som_2,
                                          values_som_1=p_list_som_1[i],
                                          values_som_2=p_list_som_2[i],
                                          epochs=epochs_on_batch)
            if print_output:
                cm_elapsed_time += time.time() - tmp

            if after_batch_callback is not None:
                after_batch_callback()

        if print_output:
            print(f'Time spend on the Components: \nSom: {som_elapsed_time} s | Convolution Matrix: {cm_elapsed_time} s')
