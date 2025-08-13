import torch
import torch.nn as nn
import re 
from collections import OrderedDict
import operator
from itertools import accumulate
import numpy as np

from .utils import get_class


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
    def __init__(self, config, dim=1, mode='cat'):
        super().__init__()
        self._branches = []
        self._dim = dim
        self.mode = mode
        for layer_name, layer_conf in config.items():
            model = get_class(layer_conf['model'])
            model = model(**layer_conf['kwargs'])
            setattr(self, layer_name, model)    # The order of self.children() will be the order in which they were added 
            self._branches.append([model, layer_name])

    def forward(self, x):
        outputs = []
        for model, _ in self._branches:
            outputs.append(model(x))

        # Now concatenate
        if self.mode == 'cat':
            return torch.cat(outputs, dim=self._dim)
        else:
            return sum(outputs)
    
    def __getitem__(self, key):
        return self._branches[key][0]
    
    def __setitem__(self, key, value):
        self._branches[key][0] = value
        setattr(self, self._branches[key][1], value)
    
    def __len__(self):
        return len(self._branches)


def hook_factory(x, surrogate, surrogate_is_grad=False):
    if surrogate_is_grad:
        def hook(grad):
            x.backward(surrogate(x) * grad) # Only makes sense in the case a function is applied element-wise on x (Jacobian is then a diagonal)

    else:
        surrogate_output = surrogate(x)

        def hook(grad):
            surrogate_output.backward(grad)
    
    return hook


class SurrogateNet(nn.Module):
    """
    Use one net for the forward pass, and another for the backward 
    """
    def __init__(self, config, clone_model=True, surrogate_is_grad=False):
        nn.Module.__init__(self)
        
        for (_, layer_conf), layer_name in zip(list(config.items())[:2], ['main', 'surrogate']):   # Only two layers are processed 
            model = get_class(layer_conf['model'])
            model = model(**layer_conf['kwargs'])
            setattr(self, layer_name, model)    # The first model is the forward, the second is the surrogate 
            
        if clone_model:
            clone(self.main, self.surrogate)

        self.surrogate_is_grad = surrogate_is_grad
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

        with torch.no_grad():
            out = self.main(x).requires_grad_(True)
        
        hook = hook_factory(x, self.surrogate, self.surrogate_is_grad)
        out.register_hook(hook)
        
        return out


def unfold(tensor, kernel_size, stride, dilation):
    """
    Unfold a tensor's last dimensions using a kernel with kernel_size and stride (both are tuples)
    """
    # To better understand this code, think that the spacial dimensions of the input tensor are stored in memory linearly as a flattened array 
    # Then, to get the patches with kernel_size, we have to convert the kernel_size shape into a non-contiguous flattened kernel 
    n_dims = len(kernel_size)   # Number of dimensions to unfold 

    volumes = list(accumulate((1,) + tensor.shape[:0:-1], operator.mul))[::-1]  # Volumes of each dimension 

    final_shape = list(tensor.shape[:-n_dims])  # Keep non-spacial shapes
    final_shape.extend((-1,) + kernel_size)     # Leave the number of kernel positions blank (-1)

    final_strides = volumes[:-n_dims]           # For the non-spacial dimensions, the strides are the volumes 
    final_strides.append(1)                     # First move the kernel by 1 element on the linearized memory 
    final_strides.extend([x * y for x, y in zip(volumes[-n_dims:], dilation)])

    size_per_dim = [(x-1) * y for x, y in zip(final_shape[-n_dims:], final_strides[-n_dims:])]  # This is how many memory elements each kernel dimension needs 
    kernel_memory = sum(size_per_dim) + 1   # This is how many memory elements the kernel spans in total (non-contiguous)

    length = volumes[-n_dims - 1] - kernel_memory + 1     # Simple formula for the flattened kernel 

    final_shape[-n_dims - 1] = length

    unfolded_tensor = tensor.as_strided(final_shape, final_strides)

    # We now remove the positions corresponding to the kernels wrapping around the edges 
    start_indexes = np.arange(0, length)
    valid_indexes = np.ones(length)
    for dim in range(-1, -n_dims, -1):
        valid_indexes = np.logical_and(valid_indexes, (start_indexes % volumes[dim - 1]) + size_per_dim[dim] + 1 <= volumes[dim - 1])

    # Finally, we apply the 'convolution' stride
    for dim in range(-1, -n_dims - 1, -1):
        valid_indexes = np.logical_and(valid_indexes, ((start_indexes % volumes[dim - 1]) // volumes[dim]) % stride[dim] == 0)
    
    indexing = [slice(None)] * unfolded_tensor.ndim 
    indexing[-n_dims - 1] = valid_indexes
    
    return unfolded_tensor[tuple(indexing)]


class ConvLike:
    """
    Provides helper methods to unfold tensors into a way compatible with some models such as GRUs
    """
    def __init__(self, in_channels, kernel_size, stride=1, dilation=1):
        self.in_channels = in_channels
        self.kernel_size = (kernel_size,) if type(kernel_size) is int else kernel_size
        self.ndim = len(self.kernel_size)
        self.stride = (stride,) * self.ndim if type(stride) is int else stride
        self.dilation = (dilation,) * self.ndim if type(dilation) is int else dilation
        self.effective_size = tuple((k - 1) * d + 1 for k, d in zip(self.kernel_size, self.dilation))

    def unfold(self, x):
        x = unfold(x, self.kernel_size, self.stride, self.dilation)
        x = torch.transpose(x, -self.ndim - 1, -self.ndim - 2)  # Transpose the input channel dimension with the sequence index dimension  
        x = x.flatten(start_dim=-self.ndim - 1, end_dim=-1) # Flatten the input channel dimensions along with the spacial ones
        return x
    
    def forward(self, x):
        new_sizes = tuple((shape - e) // s + 1 for shape, e, s in zip(x.shape[-self.ndim:], self.effective_size, self.stride))
        x = self.unfold(x)
        x = super().forward(x)  # Call the forward method of the class above this one in the MRO
        x = torch.transpose(x, -1, -2) # Put the channel dimension back in it's place 
        return x.reshape(x.shape[:-1] + new_sizes)
    

def get_padding(module, padding=None, ndim=1):
    """
    Computes the padding necessary to have an output of L / s, where L is the sequence size and s is the stride, of a convolution layer, assuming L is divisible by s. A negative right padding means that for the last stride the kernel doesn't cover the entire input
    """
    if hasattr(module, 'get_padding'):
        return module.get_padding(padding)
    
    elif hasattr(module, 'kernel_size') and hasattr(module, 'stride') and hasattr(module, 'dilation'):
        kernel_size = module.kernel_size
        stride = module.stride
        dilation = module.dilation
        effective_size = tuple((k - 1) * d + 1 for k, d in zip(kernel_size, dilation))
        return tuple(
            p for e_size, stride in zip(effective_size, stride) 
                for p in ((e_size - 1) // 2, e_size - (e_size - 1) // 2 - stride)    # Left and right padding for each dimension 
            )

    else:
        return (0,) * (2 * ndim)


def get_stride(module, ndim=1):
    if hasattr(module, 'get_stride'):
        return module.get_stride()
    
    elif hasattr(module, 'stride'):
        return module.stride
    
    else:
        return (1,) * ndim


class SequentialConv(SequentialNet):
    """
    A SequentialNet with special methods for a convolutional structure
    """
    def __init__(self, *args, ndim=1, **kwargs):
        SequentialNet.__init__(self, *args, **kwargs)
        self.ndim = ndim

    def get_padding(self, padding=None):
        if padding is None:
            padding = (0,) * (2 * self.ndim)

        layer_list = list(self.children())

        for layer in layer_list[::-1]:
            own_padding = get_padding(layer, padding, self.ndim)
            stride = get_stride(layer, self.ndim)
            padding = tuple(
                p for dim_idx, (l, r) in enumerate(zip(padding[::2], padding[1::2]))
                    for p in (
                        l * stride[dim_idx] + own_padding[2 * dim_idx],
                        r * stride[dim_idx] + own_padding[2 * dim_idx + 1]
                    )
                )

        return padding
    
    def get_stride(self):
        stride = (1,) * self.ndim
        for layer in self.children():
            own_stride = get_stride(layer)
            stride = tuple(x * y for x, y in zip(stride, own_stride))
        
        return stride
    

class ParallelConv(ParallelNet):
    def __init__(self, *args, ndim=1, **kwargs):
        ParallelNet.__init__(self, *args, **kwargs)
        self.ndim = ndim
        self.padding_list = []
        self.padding = None
        self.padding_slices = []

    def get_padding(self, padding=None):
        for layer in self.children():
            own_padding = get_padding(layer, padding=None, ndim=self.ndim)
            self.padding_list.append(own_padding)
    
        self.padding = (max([x[0] for x in self.padding_list]), max([x[1] for x in self.padding_list]))

        for branch_padding in self.padding_list:
            slice_list = [...]
            for dim_idx in range(len(branch_padding) // 2):
                slice_list.append(slice(self.padding[0] - branch_padding[2 * dim_idx], -(self.padding[1] - branch_padding[2 * dim_idx + 1]) or None))
            
            self.padding_slices.append(slice_list)

        return self.padding
    
    def get_stride(self):
        return get_stride(self._branches[0][0])
    
    def forward(self, x):
        outputs = []
        for idx, (model, _) in enumerate(self._branches):
            outputs.append(model(x[tuple(self.padding_slices[idx]])))

        # Now concatenate
        if self.mode == 'cat':
            return torch.cat(outputs, dim=self._dim)
        else:
            return sum(outputs)


class SurrogateConv(SurrogateNet):
    def __init__(self, *args, ndim=1, **kwargs):
        SurrogateNet.__init__(self, *args, **kwargs)
        self.ndim = ndim
    
    def get_padding(self, padding=None):
        get_padding(self.surrogate, padding=None, ndim=self.ndim)
        return get_padding(self.main, padding=None, ndim=self.ndim)
    
    def get_stride(self):
        return get_stride(self.main, self.ndim)
            
    
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
