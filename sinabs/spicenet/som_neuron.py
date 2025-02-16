from pathlib import Path
from torch import nn
import torch
from sinabs.from_torch import from_model

class SpiceSOMNeuron(nn.Module):
    """Run value(s) through a SPICEnet SOM neuron trained as an SNN.

    Parameters:
        standard_deviation: The width of the tuning curve (the Normal distribution).
        preferred_value: The "expected value" of the Normal distribution representing the activation function.
    """

    def __init__(
        self,
        standard_deviation: float,
        preferred_value: float,
        timesteps: int,
    ):
        super().__init__()
        self.standard_deviation = standard_deviation
        self.preferred_value = preferred_value
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
        ann.load_state_dict(torch.load((Path(__file__).parent / "som_neuron_ann.pth").resolve()))
        
        self.som_snn = [from_model(ann, input_shape=(3,), add_spiking_output=False, synops=False, num_timesteps=self.timesteps)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Check input is the correct shape
        if input.dim() != 1:
            raise ValueError(f"Input must be 1D, got {input.dim()}D")
        
        # Make std and prefered value the same shape as the input
        standard_deviation = torch.full_like(input, self.standard_deviation)
        preferred_value = torch.full_like(input, self.preferred_value)
        
        # stack the input, std and prefered value
        input = torch.stack((input, preferred_value, standard_deviation), dim=1)
        
        # Repeat the input for the number of timesteps
        input = input.repeat(self.timesteps, 1, 1)
        
        # Swap axes
        # input = torch.swapaxes(input=input, axis0=0, axis1=1)
        # print(input.shape)
        
        # Run input through the SOM SNN
        som_result = self.som_snn[0](input)
        
        return som_result[50:, :, :].mean(dim=0)