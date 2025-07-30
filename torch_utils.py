import torch
import torch.nn as nn
import re 
from collections import OrderedDict

from utils import get_class


class Freeze:
    """
    Subclass this to enable parameter freezing
    """
    def __init__(self):
        self.mask_dict = {name: torch.zeros(parameter.shape, dtype=bool) for name, parameter in self.named_parameters()}

    def freeze(self, name, mask):
        """
        Freezes (sets to zero and deactivate the gradients) the weights indicated by a mask
        """

        param_dict = dict(self.named_parameters())

        with torch.no_grad():
            self.mask_dict[name] = torch.logical_or(torch.as_tensor(mask), self.mask_dict[name])
            param_dict[name][self.mask_dict[name]] = 0

        def zero_grad(grad):
            # Use mask to zero-out the gradient without in-place modification 
            mask = torch.ones_like(grad)
            mask[self.mask_dict[name]] = 0
            return grad * mask

        # If freezing is done within torch.no_grad, then no hook is created 
        if param_dict[name].requires_grad:
            param_dict[name].register_hook(zero_grad)
            

class SequentialNet(nn.Sequential):
    """
    Creates a torch.nn.Sequential network using a config dict as input
    """
    def __init__(self, config):
        layer_dict = OrderedDict()
        for layer_name, layer_conf in config.items():
            model = get_class(layer_conf['model'])
            model = model(**layer_conf['kwargs'])
            layer_dict[layer_name] = model
            
        super().__init__(layer_dict)
        
        
class ParallelNet(nn.Module):
    """
    Concatenate the outputs of multiple models at the desired dimension 
    """
    def __init__(self, config, dim=1):
        super().__init__()
        self._branches = []
        self._dim = dim
        for layer_name, layer_conf in config.items():
            model = get_class(layer_conf['model'])
            model = model(**layer_conf['kwargs'])
            setattr(self, layer_name, model)    # The order of self.children() will be the order in which they were added 
            self._branches.append([model, layer_name])

    def forward(self, x):
        outputs = []
        for model, _ in self._branches:
            outputs.append(model.forward(x))

        # Now concatenate
        return torch.cat(outputs, dim=self._dim)
    
    def __getitem__(self, key):
        return self._branches[key][0]
    
    def __setitem__(self, key, value):
        self._branches[key][0] = value
        setattr(self, self._branches[key][1], value)
    
    def __len__(self):
        return len(self._branches)


class SurrogateNet(nn.Module):
    """
    Use one net for the forward pass, and another for the backward 
    """
    def __init__(self, config, clone_model=True):
        nn.Module.__init__(self)
        
        for (_, layer_conf), layer_name in zip(list(config.items())[:2], ['main', 'surrogate']):   # Only two layers are processed 
            model = get_class(layer_conf['model'])
            model = model(**layer_conf['kwargs'])
            setattr(self, layer_name, model)    # The first model is the forward, the second is the surrogate 
            
        if clone_model:
            clone(self.main, self.surrogate)

        self.surrogate_output = None
        
    def forward(self, x):
        # Forward pass: ("|" means a gradient block)
        # ...--> x --> main -->| out --> ...
        #        \
        #          --> surr -->

        # Backward pass:
        # ...<-- x <-- main <--| out <-- ...
        #        \              /  (connection created through hook)
        #          <-- surr <--

        self.surrogate_output = self.surrogate(x)
        with torch.no_grad():
            out = self.main(x).requires_grad_(True)
        
        def hook(grad):
            self.surrogate_output.backward(grad)
        
        out.register_hook(hook)
        
        return out


# Used to process the parameters of a torch.nn.Module
def process_parameters(module, config, device='cpu', accumulate=False):
    """
    Used to process the parameters of a torch.nn.Module, such as running them through a regularization or initialization function
    Each item in config is a dictionary of the format {'pattern': str, 'func': str, 'kwargs': dict()}  
    """
    acc = torch.tensor(0.).to(device)

    for name, param in module.named_parameters():
        for config_dict in config:
            pattern = config_dict['pattern']
            func = get_class(config_dict['func'])
            kwargs = config_dict['kwargs']
            if re.match(pattern, name):
                if accumulate:
                    acc += func(param=param, **kwargs)
                else:
                    func(param=param, **kwargs)

    if accumulate:
        return acc


def parameters(module, name, value=None):
    name_list = name.split('.')
    submodule = module
    for next_name in name_list[:-1]:
        submodule = getattr(submodule, next_name)

    if value is not None:
        setattr(submodule, name_list[-1], value)

    else:
        return getattr(submodule, name_list[-1])


def clone(modelA, modelB):
    for name, param in modelA.named_parameters():
        parameters(modelB, name, param)


def entropy_reg(param, factor=0.1):
    param = torch.abs(param)
    param = param / torch.sum(param)

    loss = -torch.sum(param * torch.log2(param))
    
    return factor * loss


def l1_reg(param, factor=0.1):
    return factor * torch.sum(torch.abs(param))
