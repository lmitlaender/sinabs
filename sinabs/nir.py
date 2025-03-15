from functools import partial
from typing import Optional, Tuple, Union

import nir
import nirtorch
import numpy as np
import torch
from torch import nn

import sinabs.layers as sl
import sinabs.spicenet as sn


def _as_pair(x) -> Tuple[int, int]:
    try:
        if len(x) == 1:
            return (x[0], x[0])
        elif len(x) >= 2:
            return tuple(x)
        else:
            raise ValueError()
    except TypeError:
        return x, x


def _import_sinabs_module(
    node: nir.NIRNode, batch_size: int, num_timesteps: int
) -> torch.nn.Module:
    if isinstance(node, nir.Affine):
        linear = nn.Linear(
            in_features=node.weight.shape[1],
            out_features=node.weight.shape[0],
            bias=True,
        )
        linear.weight.data = torch.tensor(node.weight).float()
        linear.bias.data = torch.tensor(node.bias).float()
        return linear

    elif isinstance(node, nir.Conv1d):
        conv = nn.Conv1d(
            in_channels=node.weight.shape[1],
            out_channels=node.weight.shape[0],
            kernel_size=node.weight.shape[2:],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
            bias=True,
        )
        conv.weight.data = torch.tensor(node.weight).float()
        conv.bias.data = torch.tensor(node.bias).float()
        return conv

    elif isinstance(node, nir.Conv2d):
        conv = nn.Conv2d(
            in_channels=node.weight.shape[1],
            out_channels=node.weight.shape[0],
            kernel_size=node.weight.shape[2:],
            stride=node.stride,
            padding=node.padding,
            dilation=node.dilation,
            groups=node.groups,
            bias=True,
        )
        conv.weight.data = torch.tensor(node.weight).float()
        conv.bias.data = torch.tensor(node.bias).float()
        return conv

    elif isinstance(node, nir.LI):
        if node.v_leak.shape == torch.Size([]):
            node.v_leak = node.v_leak.unsqueeze(0)
        if node.r.shape == torch.Size([]):
            node.r = node.r.unsqueeze(0)
        if any(node.v_leak != 0):
            raise ValueError("`v_leak` must be 0")
        if any(node.r != 1):
            raise ValueError("`r` must be 1")
        # TODO check for norm_input
        return sl.ExpLeakSqueeze(
            tau_mem=node.tau,
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            norm_input=False,
        )

    elif isinstance(node, nir.IF):
        return sl.IAFSqueeze(
            min_v_mem=-node.v_threshold,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            spike_threshold=node.v_threshold,
        )

    elif isinstance(node, nir.LIF):
        if node.v_leak.shape == torch.Size([]):
            node.v_leak = node.v_leak.unsqueeze(0)
        if any(node.v_leak) != 0:
            raise ValueError("`v_leak` must be 0")
        # TODO check for norm_input
        return sl.LIFSqueeze(
            tau_mem=node.tau,
            min_v_mem=None,
            num_timesteps=num_timesteps,
            batch_size=batch_size,
            spike_threshold=node.v_threshold,
            tau_syn=None,
            norm_input=False,
        )
    elif isinstance(node, nir.SumPool2d):
        return sl.SumPool2d(
            kernel_size=tuple(node.kernel_size), stride=tuple(node.stride)
        )
    elif isinstance(node, nir.Flatten):
        start_dim = node.start_dim + 1 if node.start_dim >= 0 else node.start_dim
        end_dim = node.end_dim + 1 if node.end_dim >= 0 else node.end_dim
        return nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    elif isinstance(node, nir.SPICEnetSOMNeuron):
        return sn.SpiceSOMNeuron(
            standard_deviation=node.std.item(),
            preferred_value=node.mean.item(),
            timesteps=num_timesteps,
        )
    elif isinstance(node, nir.SPICEnetSOM):
        const_LR_interaction_kernel = 0.8
        const_LR_tuning_curve = 0.8
        
        print(node.metadata)
        
        if "lrf_tuning_curve" in node.metadata:
            if "parameters" in node.metadata["lrf_tuning_curve"]:
                if "value" in node.metadata["lrf_tuning_curve"]["parameters"]:
                    const_LR_tuning_curve = node.metadata["lrf_tuning_curve"]["parameters"]["value"]
        if "lrf_interaction_kernel" in node.metadata:
            if "parameters" in node.metadata["lrf_interaction_kernel"]:
                if "value" in node.metadata["lrf_interaction_kernel"]["parameters"]:
                    const_LR_interaction_kernel = node.metadata["lrf_interaction_kernel"]["parameters"]["value"]
        
        som = sn.SpiceSOM.from_lists(
            standard_deviation=[neuron.std.item() for neuron in node.neurons],
            preferred_value=[neuron.mean.item() for neuron in node.neurons],
            value_range_start=-1,
            value_range_end=1,
            const_LR_interaction_kernel=const_LR_interaction_kernel,
            const_LR_tuning_curve=const_LR_tuning_curve,
            timesteps=num_timesteps,
        )
        
        if "iteration" in node.metadata:
            som.set_iteration(node.metadata["iteration"])
        
        return som
    elif isinstance(node, nir.SPICEnetHCM):
        const_lrf_trust_of_new = 0.8
        const_lrf_weights = 0.8
        
        if "lrf_trust_of_new" in node.metadata:
            if "parameters" in node.metadata["lrf_trust_of_new"]:
                if "value" in node.metadata["lrf_trust_of_new"]["parameters"]:
                    const_lrf_trust_of_new = node.metadata["lrf_trust_of_new"]["parameters"]["value"]
        if "lrf_weights" in node.metadata:
            if "parameters" in node.metadata["lrf_weights"]:
                if "value" in node.metadata["lrf_weights"]["parameters"]:
                    const_lrf_weights = node.metadata["lrf_trust_of_new"]["parameters"]["value"]
        
        
        hcm = sn.SpiceHCM.from_weights(
            weights=node.weights,
            activation_bar_vector_1=node.activation_bar_vector_1,
            activation_bar_vector_2=node.activation_bar_vector_2,
            const_lrf_trust_of_new=const_lrf_trust_of_new,
            const_lrf_weights=const_lrf_weights
        )
        
        if "iteration" in node.metadata:
            hcm.set_iteration(node.metadata["iteration"])        
        
        return hcm
    elif isinstance(node, nir.SPICENet):
        # ONLY allow 2 SOMs and 1 HCM until Original Spicenet multi-HCM support is added
        assert len(node.soms) == 2
        assert len(node.hcms.values()) == 1
        return sn.SpiceNet(
            spice_som_1=_import_sinabs_module(node.soms["0"], batch_size, num_timesteps),
            spice_som_2=_import_sinabs_module(node.soms["1"], batch_size, num_timesteps),
            correlation_matrix=_import_sinabs_module(list(node.hcms.values())[0], batch_size, num_timesteps)
        )
    elif isinstance(node, nir.Input):
        return nn.Identity()
    elif isinstance(node, nir.Output):
        return nn.Identity()


def from_nir(
    node: nir.NIRNode, batch_size: int = None, num_timesteps: int = None
) -> torch.nn.Module:
    """Load a sinabs model from an NIR model.

    Args:
        node (nir.NIRNode): An NIR node/graph of the model
        batch_size (int, optional): batch size of the data that is expected to be fed to the model.Defaults to None.
        num_timesteps (int, optional): Number of time steps per data sample. Defaults to None.

    NOTE:
        `batch_size` or `num_timesteps` has to be specified for the sinabs model to be instantiated correctly.

    Returns:
        torch.nn.Module: Returns a sinabs model that is equivalent to the NIR graph specified.
    """
    return nirtorch.load(
        node,
        partial(
            _import_sinabs_module, batch_size=batch_size, num_timesteps=num_timesteps
        ),
    )


def _extend_to_shape(x: Union[torch.Tensor, float], shape: Tuple) -> torch.Tensor:
    if x.shape == shape:
        return x
    elif x.shape == (1,) or x.dim() == 0:
        return torch.ones(*shape) * x
    else:
        raise ValueError(f"Not sure how to extend {x} to shape {shape}")


def _extract_sinabs_module(module: torch.nn.Module) -> Optional[nir.NIRNode]:
    if type(module) in [sl.IAF, sl.IAFSqueeze]:
        layer_shape = module.v_mem.shape[1:]
        nir_node = nir.IF(
            r=torch.ones(*layer_shape),  # Discard batch dim
            v_threshold=_extend_to_shape(module.spike_threshold.detach(), layer_shape),
        )
        return nir_node
    elif type(module) in [sl.LIF, sl.LIFSqueeze]:
        layer_shape = module.v_mem.shape[0]
        return nir.LIF(
            tau=module.tau_mem.detach(),
            v_threshold=module.spike_threshold.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    elif type(module) in [sl.ExpLeak, sl.ExpLeakSqueeze]:
        return nir.LI(
            tau=module.tau_mem.detach(),
            v_leak=torch.zeros_like(module.tau_mem.detach()),
            r=torch.ones_like(module.tau_mem.detach()),
        )
    elif isinstance(module, torch.nn.Linear):
        if module.bias is None:  # Add zero bias if none is present
            return nir.Affine(
                module.weight.detach(), torch.zeros(*module.weight.shape[:-1])
            )
        else:
            return nir.Affine(module.weight.detach(), module.bias.detach())
    elif isinstance(module, torch.nn.Conv1d):
        return nir.Conv1d(
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=(
                module.bias.detach()
                if module.bias
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, torch.nn.Conv2d):
        return nir.Conv2d(
            input_shape=None,
            weight=module.weight.detach(),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=(
                module.bias.detach()
                if isinstance(module.bias, torch.Tensor)
                else torch.zeros((module.weight.shape[0]))
            ),
        )
    elif isinstance(module, sl.SumPool2d):
        return nir.SumPool2d(
            kernel_size=_as_pair(module.kernel_size),  # (Height, Width)
            stride=_as_pair(
                module.kernel_size if module.stride is None else module.stride
            ),  # (Height, width)
            padding=(0, 0),  # (Height, width)
        )
    elif isinstance(module, nn.Flatten):
        # Getting rid of the batch dimension for NIR
        start_dim = module.start_dim - 1 if module.start_dim > 0 else module.start_dim
        end_dim = module.end_dim - 1 if module.end_dim > 0 else module.end_dim
        return nir.Flatten(
            input_type=None,
            start_dim=start_dim,
            end_dim=end_dim,
        )
    elif isinstance(module, sn.SpiceSOMNeuron):
        return nir.SPICEnetSOMNeuron(
            std=np.array(module.standard_deviation),
            mean=np.array(module.preferred_value),
        )
    elif isinstance(module, sn.SpiceSOM):
        return nir.SPICEnetSOM(
            neurons=[
                nir.SPICEnetSOMNeuron(std=np.array(module.standard_deviation[i]), mean=np.array(module.preferred_value[i]))
                for i in range(len(module.standard_deviation))
            ],
            metadata={"iteration": module.get_iteration(), "lrf_interaction_kernel": {"type": "const", "parameters": {"value": module.const_LR_interaction_kernel}}, "lrf_tuning_curve": {"type": "const", "parameters": {"value": module.const_LR_tuning_curve}}}
        )
    elif isinstance(module, sn.SpiceHCM):
        return nir.SPICEnetHCM(
            weights=module.weights,
            activation_bar_vector_1=module.activation_bar_vector_1,
            activation_bar_vector_2=module.activation_bar_vector_2,
            metadata={"iteration": module.get_iteration(), "lrf_trust_of_new": {"type": "const", "parameters": {"value": module.const_lrf_trust_of_new}}, "lrf_weights": {"type": "const", "parameters": {"value": module.const_lrf_weights}}}
        )
    elif isinstance(module, sn.SpiceNet):
        return nir.SPICENet.from_lists(soms=[_extract_sinabs_module(module.som_1[0]), _extract_sinabs_module(module.som_2[0])], hcms=[(0, 1, _extract_sinabs_module(module.get_correlation_matrix()))])
    print(f"Module {module} not supported")
    raise NotImplementedError(f"Module {type(module)} not supported")


def to_nir(
    module: torch.nn.Module, sample_data: torch.Tensor, model_name: str = "model"
) -> nir.NIRNode:
    """Generate a NIRGraph given a sinabs model.

    Args:
        module (torch.nn.Module): The sinabs model to be converted to NIR graph
        sample_data (torch.Tensor): A sample data that can be used to extract various shapes and internal states.
        model_name (str, optional): The name of the top level model. Defaults to "model".

    Returns:
        nir.NIRNode: Returns the equivalent NIR object.
    """
    return nirtorch.extract_nir_graph(
        module,
        _extract_sinabs_module,
        sample_data,
        model_name=model_name,
        ignore_dims=[0],
    )
