from pathlib import Path
from torch import nn
import torch
from sinabs.from_torch import from_model
import math
import numpy as np

class SpiceSOM(nn.Module):
    """Run value(s) through a SPICEnet SOM with a single SNN acting as the SOM Neuron.
    The single SNN improves the efficiency of the SOM by running all neurons in parallel.

    Parameters:
        standard_deviation: list of standard deviations for all neurons in order of neurons.
        preferred_value: list of preferred values for all neurons in order of neurons.
        timesteps: The number of timesteps to run the SNN for.
    """
    def __init__(self,
                n_neurons: int,
                value_range_start: float,
                value_range_end: float,
                const_LR_interaction_kernel: float,
                const_LR_tuning_curve: float,
                timesteps: int,
                average_from: int | None = None):
        super().__init__()
        
        
        if average_from is None:
            average_from = math.floor(timesteps / 2)
        if average_from >= timesteps:
            raise ValueError("average_from must be less than timesteps")
        
        self.average_from = average_from
        
        self.__iteration = 0
        self.const_LR_interaction_kernel = const_LR_interaction_kernel
        self.const_LR_tuning_curve = const_LR_tuning_curve
        
        distance = value_range_end - value_range_start
        step_size = distance / n_neurons
        
        self.preferred_value = []
        self.standard_deviation = []
        
        pos = value_range_start + step_size / 2.0

        for i in range(n_neurons):
            self.preferred_value.append(pos)
            self.standard_deviation.append(0.001)
            pos += step_size
            
        self.preferred_value = self.preferred_value
        self.standard_deviation = self.standard_deviation
        
        self.timesteps = timesteps
        
        ann = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        ann.load_state_dict(torch.load((Path(__file__).parent / "low_range_log1p.pth").resolve()))
        
        self.som_snn_neuron = [from_model(ann, input_shape=(3,), add_spiking_output=False, synops=False, num_timesteps=self.timesteps)]

    @classmethod
    def from_lists(
        cls,
        standard_deviation: list[float],
        preferred_value: list[float],
        value_range_start: float,
        value_range_end: float,
        const_LR_interaction_kernel: float,
        const_LR_tuning_curve: float,
        timesteps: int,
    ):
        som = cls(len(standard_deviation), value_range_start, value_range_end, const_LR_interaction_kernel, const_LR_tuning_curve, timesteps)
        
        som.standard_deviation = standard_deviation
        som.preferred_value = preferred_value
        
        return som

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check input is the correct shape
        if input.dim() != 1:
            raise ValueError(f"Input must be 1D, got {input.dim()}D")
        
        result = []
        
        std_tensor = torch.Tensor(self.standard_deviation)
        pref_tensor = torch.Tensor(self.preferred_value)
        
        # Make std and prefered value the same shape as the input
        for value in input.tolist():
            # stack the input, std and prefered value
            repeated_value = torch.full_like(std_tensor, value)
            input = torch.stack((repeated_value, pref_tensor, std_tensor), dim=1)
            
            # Repeat the input for the number of timesteps
            input = input.repeat(self.timesteps, 1, 1)
            
            with torch.no_grad():
                subresult = self.som_snn_neuron[0](input)
            result.append(subresult)
            
        # Transform result from list of answers for each neuron to list of activation vectors of each neuron for a value
        result = torch.stack(result, dim=0)
        return torch.expm1(result[:, self.average_from:, :, :].mean(dim=1))
    
    def __len__(self) -> int:
        return len(self.standard_deviation)
    
    def print_neurons(self):
        for i, (pref_val, std) in enumerate(zip(self.preferred_value, self.standard_deviation)):
            print(f"Neuron {i}: preferred value: {pref_val}, standard deviation: {std}")
            
    
    def fit(self, values: list[float], epochs: int):
        for epoch in range(epochs):
            for i in range(len(values)):
                winning_neuron_index, _ = self.__argmax_neuron_activation(values[i])

                for j in range(len(self.preferred_value)):
                    self.update_neuron(j, values[i],
                                             self.const_LR_tuning_curve, # Only constant LR
                                             self.const_LR_interaction_kernel, # Only constant LR
                                             j - winning_neuron_index)
                self.__iteration += 1
                
    def __argmax_neuron_activation(self, value: float):
        """
        Calculates the neuron with the highest activation value.
        :param value: The value for wich the activation values should be calculated.
        :return: The index of the winning neuron
        and a dictionary containing the index of a neuron with the calculated activation value.
        """
        # Get activation values over timesteps
        results = self.forward(torch.Tensor([value]))
        
        # Get the mean activation value over timesteps
        winner_neuron_values = results[0, :, 0]
        
        # Get the index of the winning neuron
        winner_neuron_index = torch.argmax(winner_neuron_values).item()
        activation_dict = {i: winner_neuron_values[i].item() for i in range(len(winner_neuron_values))}
        
        return winner_neuron_index, activation_dict
        
    def update_neuron(self, index: int, value: float, learn_rate: float, interaction_kernel_learning_rate: float,
                distance_to_winner: int):
        """
        Updates the weights of the neurons.
        :param index: index of the neuron to update
        :param value: The value in wich "direction" the neuron should move.
        :param learn_rate: The learn rate. (This value should be depending on the iteration.)
        :param interaction_kernel_learning_rate: The sigma of the interaction kernel.
        (This value should be depending on the iteration.)
        :param distance_to_winner: The distance between this neuron and the winning neuron, in the SOM.
        Here are not the weights / postions in the value range relevant.
        (N1 weight: 31.9) --- (N2 weight: 32) --- (N3 weight: 32.4) --- (N4 weight: 33)
        The Distance of N4 to N1 is 3
        :return:
        """
        interaction_kernel_value = math.exp(
            (-abs(distance_to_winner) ** 2) / (2 * interaction_kernel_learning_rate ** 2))
        self.preferred_value[index] += learn_rate * interaction_kernel_value * (value - self.preferred_value[index])
        self.standard_deviation[index] += learn_rate * interaction_kernel_value * (
                (value - self.preferred_value[index]) ** 2 - self.standard_deviation[index] ** 2
        )
        
    def all_activation_for_values(self, values: list[float]):
        """
        Returns the activation values for a list of values.
        :param values: The values for wich the activation has to be calculated.
        :return: An array of activation values, ordered like the input list.
        """
        return np.array([self.get_activation_vector(value) for value in values]).T

        
    def calculate_activation_values(self, values: list[float]):
        """
        Calculates activation values for all neurons in the som.
        :param values:
        :return: A numpy array the first col is the preferred value, second col is the tuning curve width
        and the following cols are the activation values.
        """
        activation_values = np.array(self.all_activation_for_values(values))
        return np.concatenate((
            np.array(
                [[pref for pref in self.preferred_value],
                 [std for std in self.standard_deviation]]).transpose(),
            activation_values),
            axis=1)

    def naive_decode(self, value: float, neuron_index: int) -> float:
        activation_value = value
        
        r = math.sqrt(2 * self.standard_deviation[neuron_index] ** 2 * math.log(
            math.sqrt(2 * math.pi) * activation_value * self.standard_deviation[neuron_index] ** 2, 10))

        if neuron_index < len(self.standard_deviation) / 2:
            return self.preferred_value[neuron_index] - r
        else:
            return self.preferred_value[neuron_index] + r
        
    def get_winning_neuron_index(self, value: float) -> tuple[int, float]:
        winning_neuron_index, activation_values = self.__argmax_neuron_activation(value)
        return winning_neuron_index, activation_values[winning_neuron_index]
    
    def get_activation_vector(self, value: float) -> np.array:
        # Get activation values over timesteps
        results = self.forward(torch.Tensor([value]))
        
        # Get the mean activation value over timesteps
        winner_neuron_values = results[0, :, 0]
        
        return np.array(winner_neuron_values.tolist())
    

    def get_as_matrix(self) -> np.ndarray:
        """
        Returns all neurons like a table with column 0 representing the preferred value
        and column 1 representing the tuning curve width.
        :return: A numpy matrix.
        """
        return np.array([self.preferred_value,
                         self.standard_deviation]).transpose()
        
    def get_iteration(self) -> int:
        return self.__iteration
    
    def set_iteration(self, iteration: int):
        self.__iteration = iteration