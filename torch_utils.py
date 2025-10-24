import torch
import torch.nn as nn
import torch.nn.functional as F
import re 
from collections import OrderedDict
import operator
from itertools import accumulate
import numpy as np
from itertools import combinations
from math import prod

from .utils import get_class, num_volterra


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
            if x.requires_grad:
                x.backward(surrogate(x) * grad, retain_graph=True) # Only makes sense in the case a function is applied element-wise on x (Jacobian is then a diagonal)

    else:
        surrogate_output = surrogate(x)

        def hook(grad):
            surrogate_output.backward(grad, retain_graph=True)
    
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
    

def get_padding(module, ndim=1):
    """
    Computes the padding necessary to have an output of L / s, where L is the sequence size and s is the stride, of a convolution layer, assuming L is divisible by s. A negative right padding means that for the last stride the kernel doesn't cover the entire input
    """
    if hasattr(module, 'get_padding'):
        return module.get_padding()
    
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

    def get_padding(self):
        padding = (0,) * (2 * self.ndim)

        layer_list = list(self.children())

        for layer in layer_list[::-1]:
            own_padding = get_padding(layer, self.ndim)
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

    def get_padding(self):
        for layer in self.children():
            own_padding = get_padding(layer, ndim=self.ndim)
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
            outputs.append(model(x[tuple(self.padding_slices[idx])]))

        # Now concatenate
        if self.mode == 'cat':
            return torch.cat(outputs, dim=self._dim)
        else:
            return sum(outputs)


class SurrogateConv(SurrogateNet):
    def __init__(self, *args, ndim=1, **kwargs):
        SurrogateNet.__init__(self, *args, **kwargs)
        self.ndim = ndim
    
    def get_padding(self):
        get_padding(self.surrogate, ndim=self.ndim)
        return get_padding(self.main, ndim=self.ndim)
    
    def get_stride(self):
        return get_stride(self.main, self.ndim)
            

class Volterra(nn.Module):
    """
    Creates the features combining different time steps. Combine this with ConvLike.
    For order=1, use Conv1d instead.
    """
    def __init__(self, order, memory, valid_indexes=None):
        self.order = order
        self.memory = memory
        if valid_indexes is not None and type(valid_indexes) is list:
            self.valid_indexes = valid_indexes
        else:
            self.valid_indexes = list(range(num_volterra(order, memory)))

        self.num_features = len(self.valid_indexes)
        nn.Module.__init__(self)

    def forward(self, x):
        # Assume x has shape (N, L, M), with N batch size, L the number of sequences, and M the memory size
        output = x.new_zeros(x.shape[:-1] + (0,), requires_grad=x.requires_grad)
        if self.memory == 0:
            return output
        
        else:
            if self.order > 1:
                for idx, c in enumerate(combinations(range(self.order + self.memory - 1), self.memory - 1)): # Stars and bars combinatorics 
                    if idx in self.valid_indexes:
                        powers = [b - a - 1 for a, b in zip((-1,) + c, c + (self.order + self.memory - 1,))]    # The difference between two bars (minus 1) is the number of elements 
                        start = False
                        for idx, power in enumerate(powers):
                            if power != 0:
                                if not start:
                                    prod = x[:, :, [idx]] ** power
                                    start = True

                                else:
                                    prod = prod * x[:, :, [idx]] ** power

                        output = torch.cat((output, prod), dim=-1)

            else:
                output = x
            
            return output
        

class ConvVolterra(ConvLike, Volterra):
    def __init__(self, in_channels, kernel_size, stride=1, dilation=1, order=1):
        Volterra.__init__(self, order=order, memory=kernel_size * in_channels)
        super().__init__(in_channels, kernel_size, stride, dilation)


class ConvGRU(ConvLike, nn.GRU, Freeze):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, num_layers=1, bias=True, dropout=0.0, dtype=None):
        nn.GRU.__init__(
            self, input_size=in_channels * kernel_size, hidden_size=out_channels, num_layers=num_layers, 
            bias=bias, batch_first=True, dropout=dropout, dtype=get_class(dtype)
        )
        super().__init__(in_channels, kernel_size, stride, dilation)
        Freeze.__init__(self)

    def forward(self, x):
        x = self.unfold(x)
        x, _ = nn.GRU.forward(self, x)
        x = torch.transpose(x, -1, -2) # Put the channel dimension back in it's place 
        return x
    

class Conv1d(nn.Conv1d, Freeze):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, device=None, dtype=None):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, bias=bias, device=device, dtype=get_class(dtype))
        Freeze.__init__(self)


class Activation(nn.Module, Freeze):
    def __init__(self, offset=0., scale=1., offset_requires_grad=False, scale_requires_grad=False):
        nn.Module.__init__(self)
        self.offset = nn.Parameter(torch.tensor(offset), requires_grad=offset_requires_grad)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=scale_requires_grad)
        Freeze.__init__(self)

    def forward(self, x):
        if self.offset.ndim == 0:
            return self.scale * self.basis(x - self.offset)
        else:
            return (self.scale * self.basis(x[..., None] - self.offset)).movedim(-1, 1).flatten(1, 2)
        

class GaussianMixturePDF(nn.Module):
    def __init__(self, dim, n_kernels):
        nn.Module.__init__(self)
        self.dim = dim
        self.n_kernels = n_kernels
        
    def forward(self, x, params):
        n_dim = x.shape[self.dim]
        
        weight, mean, var = self.separate_params(params)
        
        weight = self.reshape(x.shape, weight, False)
        mean = self.reshape(x.shape, mean, True)
        var = self.reshape(x.shape, var, False)

        return  torch.sum(
            weight * torch.exp(
                -torch.sum(torch.abs(x[..., None] - mean) ** 2, dim=self.dim) / (2 * var)
            ) / (2 * np.pi * var) ** (n_dim / 2), dim=-1
        )

    def separate_params(self, params):
        # Split parameters
        log_weight = params[..., :self.n_kernels]
        mean = params[..., self.n_kernels:-1]
        log_var = params[..., -1:]

        # Compute probabilities and variance
        weight = F.softmax(log_weight, dim=-1)
        var = torch.exp(log_var)
        
        return weight, mean, var
    
    def reshape(self, size, tensor, is_mean=True, copy=True):
        total_dim = len(size)
        if is_mean:
            shape = [1] * (total_dim + 1)
            if copy:
                shape[self.dim] = size[self.dim]
            shape[-1] = -1
            shape[0] = tensor.shape[0]
            tensor = tensor.view(*shape)
            
        else:
            shape = [1] * (total_dim)
            shape[0] = tensor.shape[0]
            shape[-1] = tensor.shape[-1]
            tensor = tensor.view(*shape)
        
        return tensor

    def sample(self, size, params):
        # Split parameters
        weight, mean, var = self.separate_params(params)
        var = self.reshape(size, var, False)

        # Number of samples
        num_samples = prod(size) // size[self.dim] // weight.shape[0]

        # Draw mixture component indices
        comp_idx = torch.multinomial(weight, num_samples=num_samples, replacement=True)
        mean = self.reshape(size, mean, True)
        comp_idx = self.reshape(size, comp_idx, True, False).expand(-1, *mean.shape[1:-1], -1)

        # Gather selected means
        chosen_means = torch.gather(mean, -1, comp_idx).transpose(self.dim, 0)
        comp_idx = comp_idx.transpose(self.dim, 0)

        # Sample from corresponding Gaussian
        eps = torch.randn_like(chosen_means)
        samples = chosen_means + eps * torch.sqrt(var)
        
        # Reshape
        samples = samples.reshape(size[self.dim], *size[:self.dim], *size[self.dim+1:]).transpose(self.dim, 0)
        comp_idx = comp_idx.reshape(size[self.dim], *size[:self.dim], *size[self.dim+1:]).transpose(self.dim, 0)

        return samples, comp_idx, weight


class BSpline(Activation):
    def __init__(self, *args, degree=1, width=1, **kwargs):
        Activation.__init__(self, *args, **kwargs)
        self.degree = degree
        self.width = width
        self.register_buffer('helper_points', torch.linspace(0, self.degree * self.width, self.degree + 1))
        
    def basis(self, x):
        # x has shape (B, Cin, N)
        
        # Read the Wikipedia article on B-splines using Cox-de Boor to understand the following
        # We start by implementing x - t_i, and by our definition B_i,0(x) = {if 0 <= x - t_i < 1: 1, otherwise: 0}
        # B_0,p(x) has a maxima at (self.degree + 1) * self.width / 2, thus we offset all points for the maxima to coincide with grid
        # Float point precision may impact the result for x coinciding with the grid (x >= 0, x < self.width is not computed accurately everytime)
        x = x + (self.degree + 1) * self.width / 2
        x = x[..., None] - self.helper_points  # (B, Cin, N, G)
        bn = torch.logical_and(x >= 0, x < self.width)

        # Cox-de Boor formula
        for k in range(self.degree):
            bn = (x[..., :-(k + 1)] * bn[..., :-1] + ((k + 2) * self.width - x[..., :-(k + 1)]) * bn[..., 1:]) / ((k + 1) * self.width)  # The grid dimension gets smaller by one at each iteration

        # Attention: for degree = 0, this function has no gradient
        return bn.flatten(start_dim=-2)

    
class Step(Activation):
    def __init__(self, *args, **kwargs):
        Activation.__init__(self, *args, **kwargs)

    def basis(self, x):
        return (x >= 0).to(dtype=x.dtype)


class ApproximateStep(Activation):
    def __init__(self, width=1, *args, **kwargs):
        Activation.__init__(self, *args, **kwargs)
        self.width = width / 2

    def basis(self, x):
        return 1 / 2 * torch.logical_and(x >= -self.width, x < 0) * (x ** 2 / self.width ** 2 + 2 * x / self.width + 1) \
            + 1 / 2 * torch.logical_and(x >= 0, x < self.width) * (-x ** 2 / self.width ** 2 + 2 * x / self.width + 1) \
            + 1. * (x >= self.width)


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


# Computes the number of bytes used by the model 
def memory_size(model, keywords=('weight', 'bias')):
    memory = {keyword: 0 for keyword in keywords}
    memory['other'] = 0
    for layer in model.children():
        for name, param in layer.named_parameters():
            for keyword in keywords:
                if keyword in name:
                    memory[keyword] += param.numel() * param.element_size()
                else:
                    memory['other'] += param.numel() * param.element_size()

    return memory


def entropy_reg(param, factor=0.1):
    param = torch.abs(param)
    param = param / torch.sum(param)

    loss = -torch.sum(param * torch.log2(param))
    
    return factor * loss


def l1_reg(param, factor=0.1):
    return factor * torch.sum(torch.abs(param))


def get_layers(**kwargs):
    layers = OrderedDict()

    if kwargs['model'] == 'volterra':
        """
        Parameters:
        'n_orders':
        'in_channels':
        'out_channels':
        'order_[i]_size':
        'stride':
        """
        volterra = OrderedDict()
        for order in range(1, kwargs['n_orders'] + 1):
            volterra[f'order_{order}'] = {
                'model': 'torch_utils.ConvVolterra',
                'kwargs': {
                    'in_channels': kwargs['in_channels'],
                    'kernel_size': kwargs[f'order_{order}_size'],
                    'stride': kwargs['stride'],
                    'order': order
                }
            }
        layers['layer_1'] = {
            'model': 'torch_utils.ParallelConv',
            'kwargs': {
                'config': volterra,
            }
        }
        layers['layer_2'] = {
            'model': 'torch_utils.Conv1d',   # You can also use 'torch.nn.Conv1d', but this version also has a cost function
            'kwargs': {
                'in_channels': sum([num_volterra(order, kwargs[f'order_{order}_size'] * kwargs['in_channels']) for order in range(1, kwargs['n_orders'] + 1)]),
                'out_channels': kwargs['out_channels'],
                'kernel_size': 1,   # kernel_size and stride are set to one since we just want to sum across the channel dimensions 
                'stride': 1
            }
        }
        
    elif kwargs['model'] == 'multi_volterra':
        """
        Parameters:
        'n_layers':
        'in_channels':
        'out_channels':
        'n_orders_[i]':
        'hidden_channels_[i]':
        'order_[j]_size_[i]':
        'stride_[i]':
        """
        for layer_idx in range(1, kwargs['n_layers'] + 1):
            layers[f'layer_{layer_idx}'] = {
                'model': 'torch_utils.SequentialConv',
                'kwargs': {
                    'config': get_layers(
                        model='volterra',
                        **{
                            'n_orders': kwargs[f'n_orders_{layer_idx}'],
                            'in_channels': kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                            'out_channels': kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                            'stride': kwargs[f'stride_{layer_idx}'],
                            **{f'order_{order}_size': kwargs[f'order_{order}_size_{layer_idx}'] for order in range(1, kwargs[f'n_orders_{layer_idx}'] + 1)}
                        }
                    )
                }
            }
        
    elif kwargs['model'] == 'gru':
        """
        Parameters:
        'n_layers':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        'num_layers_[i]':
        """
        for gru_idx in range(1, kwargs['n_layers'] + 1):
            layers[f'layer_{gru_idx}'] = {
                'model': 'torch_utils.ConvGRU',
                'kwargs': {
                    'in_channels': kwargs['in_channels'] if gru_idx == 1 else kwargs[f'hidden_channels_{gru_idx - 1}'],
                    'out_channels': kwargs[f'hidden_channels_{gru_idx}'],
                    'kernel_size': kwargs[f'kernel_size_{gru_idx}'],
                    'stride': kwargs[f'stride_{gru_idx}'],
                    'num_layers': kwargs[f'num_layers_{gru_idx}']
                }
            }
        layers[f'layer_{kwargs["n_layers"] + 1}'] = {
            'model': 'torch_utils.Conv1d',
            'kwargs': {
                'in_channels': kwargs[f'hidden_channels_{kwargs["n_layers"]}'],
                'out_channels': kwargs['out_channels'],
                'kernel_size': 1,
                'stride': 1,
            }
        }
        
    elif kwargs['model'] == 'conv':
        """
        Parameters:
        'n_layers':
        'in_channels':
        'out_channels':
        'activation':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        layers['layer_1'] = {
            'model': 'torch_utils.Conv1d',
            'kwargs': {
                'in_channels': kwargs['in_channels'],
                'out_channels': kwargs['hidden_channels_1'],
                'kernel_size': kwargs['kernel_size_1'],
                'stride': kwargs['stride_1'],
            }
        }
        for layer_idx in range(1, kwargs['n_layers']):
            layers[f'layer_{layer_idx}_act'] = {
                'model': kwargs['activation'],
                'kwargs': {}
            }
            layers[f'layer_{layer_idx + 1}'] = {
                'model': 'torch_utils.Conv1d',
                'kwargs': {
                    'in_channels': kwargs[f'hidden_channels_{layer_idx}'],
                    'out_channels': kwargs['out_channels'] if layer_idx == kwargs['n_layers'] - 1 else kwargs[f'hidden_channels_{layer_idx + 1}'],
                    'kernel_size': kwargs[f'kernel_size_{layer_idx + 1}'],
                    'stride': kwargs[f'stride_{layer_idx + 1}'],
                }
            }
        
    elif kwargs['model'] == 'spline_kan':
        """
        Parameters:
        'grid_size':
        'grid_min':
        'grid_max':
        'degree':
        'width':
        'in_channels':
        'out_channels':
        'kernel_size':
        'stride':
        """
        branches = OrderedDict()
        for grid_idx in range(kwargs['grid_size']):
            branch_layers = OrderedDict()
            branch_layers[f'layer_1'] = {
                'model': 'torch_utils.BSpline',
                'kwargs': {
                    'offset': kwargs['grid_min'] + grid_idx / (kwargs['grid_size'] - 1) * (kwargs['grid_max'] - kwargs['grid_min']),
                    'degree': kwargs['degree'],
                    'width': kwargs['width']
                }
            }
            branch_layers[f'layer_2'] = {
                'model': 'torch_utils.Conv1d',
                'kwargs': {
                    'in_channels': kwargs['in_channels'],
                    'out_channels': kwargs['out_channels'],
                    'kernel_size': kwargs['kernel_size'],
                    'stride': kwargs['stride']
                }
            }
            branches[f'point_{grid_idx}'] = {
                'model': 'torch_utils.SequentialConv',
                'kwargs': {'config': branch_layers}
            }
        layers['layer_1'] = {
            'model': 'torch_utils.ParallelConv',
            'kwargs': {
                'config': branches,
                'mode': 'sum',
            }
        }
        
    elif kwargs['model'] == 'multi_spline_kan':
        """
        Parameters:
        'n_layers':
        'grid_size':
        'grid_min':
        'grid_max':
        'width':
        'degree':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        for layer_idx in range(1, kwargs['n_layers'] + 1):
            layers[f'layer_{layer_idx}'] = get_layers(
                model='spline_kan',
                grid_size=kwargs['grid_size'], grid_min=kwargs['grid_min'], grid_max=kwargs['grid_max'],
                degree=kwargs['degree'],
                width=kwargs['width'],
                in_channels=kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                out_channels=kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                kernel_size=kwargs[f'kernel_size_{layer_idx}'],
                stride=kwargs[f'stride_{layer_idx}'],
            )['layer_1']

    elif kwargs['model'] == 'multi_default_kan':
        """
        Parameters:
        'n_layers':
        'grid_size':
        'grid_min':
        'grid_max':
        'width':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        for layer_idx in range(1, kwargs['n_layers'] + 1):
            parallel = OrderedDict()
            sequential = OrderedDict()
            parallel['layer_1'] = get_layers(
                model='spline_kan',
                grid_size=kwargs['grid_size'], grid_min=kwargs['grid_min'], grid_max=kwargs['grid_max'],
                degree=2,
                width=kwargs['width'],
                in_channels=kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                out_channels=kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                kernel_size=kwargs[f'kernel_size_{layer_idx}'],
                stride=kwargs[f'stride_{layer_idx}'],
            )['layer_1']
            sequential['layer_1'] = {
                'model': 'torch.nn.SiLU',
                'kwargs': {}
            }
            sequential['layer_2'] = {
                'model': 'torch_utils.Conv1d',
                'kwargs': {
                    'in_channels': kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                    'out_channels': kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                    'kernel_size': kwargs[f'kernel_size_{layer_idx}'],
                    'stride': kwargs[f'stride_{layer_idx}'],
                }
            }
            parallel['layer_2'] = {
                'model': 'torch_utils.SequentialConv',
                'kwargs': {'config': sequential}
            }
            layers[f'layer_{layer_idx}'] = {
                'model': 'torch_utils.ParallelConv',
                'kwargs': {
                    'config': parallel,
                    'mode': sum,
                }
            }
    
    elif kwargs['model'] == 'res_relu':
        """
        Parameters:
        'in_channels':
        'out_channels':
        'kernel_size':
        'stride':
        """
        branches = OrderedDict()
        branch_1 = OrderedDict()
        branch_2 = OrderedDict()
        branch_1['layer_1'] = {
            'model': 'torch.nn.Identity',
            'kwargs': {}
        }
        branch_1['layer_2'] = {
            'model': 'torch_utils.Conv1d',
            'kwargs': {
                'in_channels': kwargs['in_channels'],
                'out_channels': kwargs['out_channels'],
                'kernel_size': kwargs['kernel_size'],
                'stride': kwargs['stride']
            }
        }
        branch_2['layer_1'] = {
            'model': 'torch.nn.ReLU',
            'kwargs': {}
        }
        branch_2['layer_2'] = {
            'model': 'torch_utils.Conv1d',
            'kwargs': {
                'in_channels': kwargs['in_channels'],
                'out_channels': kwargs['out_channels'],
                'kernel_size': kwargs['kernel_size'],
                'stride': kwargs['stride']
            }
        }
        branches['layer_1'] = {
            'model': 'torch_utils.SequentialConv',
            'kwargs': {'config': branch_1}
        }
        branches['layer_2'] = {
            'model': 'torch_utils.SequentialConv',
            'kwargs': {'config': branch_2}
        }
        layers['layer_1'] = {
            'model': 'torch_utils.ParallelConv',
            'kwargs': {
                'config': branches,
                'mode': 'sum',
            }
        }
        
    elif kwargs['model'] == 'multi_res_relu':
        """
        Parameters:
        'n_layers':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        for layer_idx in range(1, kwargs['n_layers'] + 1):
            layers[f'layer_{layer_idx}'] = get_layers(
                model='res_relu',
                in_channels=kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                out_channels=kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                kernel_size=kwargs[f'kernel_size_{layer_idx}'],
                stride=kwargs[f'stride_{layer_idx}'],
            )['layer_1']
        
    elif kwargs['model'] == 'multi_res_relu_lin':
        """
        Parameters:
        'n_layers':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        layers['layer_1'] = {
            'model': 'torch_utils.Conv1d',
            'kwargs': {
                'in_channels': kwargs['in_channels'],
                'out_channels': kwargs['hidden_channels_1'],
                'kernel_size': kwargs['kernel_size_1'],
                'stride': kwargs['stride_1']
            }
        }
        for layer_idx in range(2, kwargs['n_layers'] + 1):
            layers[f'layer_{layer_idx}'] = get_layers(
                model='res_relu',
                in_channels=kwargs[f'hidden_channels_{layer_idx - 1}'],
                kernel_size=kwargs[f'kernel_size_{layer_idx}'],
                stride=kwargs[f'stride_{layer_idx}'],
                out_channels=kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
            )['layer_1']
            
    elif kwargs['model'] == 'rect':
        """
        Parameters:
        'in_channels':
        'offset':
        'width':
        'grad_width':
        """
        rect = OrderedDict()
        surrogate = OrderedDict()
        rect['layer_1'] = {
            'model': 'torch_utils.BSpline',
            'kwargs': {
                'offset': kwargs['offset'],
                'degree': 0,
                'width': kwargs['width'],
            }
        }
        surrogate['layer_1'] = {
            'model': 'torch_utils.BSpline',
            'kwargs': {
                'offset': kwargs['offset'] - kwargs['width'] / 2,
                'degree': 1,
                'width': kwargs['grad_width'],
                'scale': 1 / kwargs['grad_width']
            }
        }
        surrogate['layer_2'] = {
            'model': 'torch_utils.BSpline',
            'kwargs': {
                'offset': kwargs['offset'] + kwargs['width'] / 2,
                'degree': 1,
                'width': kwargs['grad_width'],
                'scale': - 1 / kwargs['grad_width']
            }
        }
        rect['layer_2'] = {
            'model': 'torch_utils.ParallelConv',
            'kwargs': {'config': surrogate, 'mode': 'sum'}
        }
        layers['layer_1'] = {
            'model': 'torch_utils.SurrogateConv',
            'kwargs': {'config': rect, 'clone_model': False, 'surrogate_is_grad': True}
        }
        
    elif kwargs['model'] == 'rect_kan':
        """
        Parameters:
        'grid_size':
        'grid_min':
        'grid_max':
        'width':
        'grad_width':
        'in_channels':
        'out_channels':
        'kernel_size':
        'stride':
        """
        branches = OrderedDict()
        for grid_idx in range(kwargs['grid_size']):
            branch_layers = OrderedDict()
            branch_layers[f'layer_1'] = get_layers(
                model='rect',
                in_channels=kwargs['in_channels'],
                offset=kwargs['grid_min'] + grid_idx / (kwargs['grid_size'] - 1) * (kwargs['grid_max'] - kwargs['grid_min']),
                width=kwargs['width'],
                grad_width=kwargs['grad_width']
            )['layer_1']      
            branch_layers[f'layer_2'] = {        
                'model': 'torch_utils.Conv1d',
                'kwargs': {
                    'in_channels': kwargs['in_channels'],
                    'out_channels': kwargs['out_channels'],
                    'kernel_size': kwargs['kernel_size'],
                    'stride': kwargs['stride']
                }
            }
            branches[f'point_{grid_idx}'] = {
                'model': 'torch_utils.SequentialConv',
                'kwargs': {'config': branch_layers}
            }
        layers['layer_1'] = {
            'model': 'torch_utils.ParallelConv',
            'kwargs': {
                'config': branches,
                'mode': 'sum',
            }
        }
        
    elif kwargs['model'] == 'multi_rect_kan':
        """
        Parameters:
        'n_layers':
        'grid_size':
        'grid_min':
        'grid_max':
        'width':
        'grad_width':
        'in_channels':
        'out_channels':
        'hidden_channels_[i]':
        'kernel_size_[i]':
        'stride_[i]':
        """
        for layer_idx in range(1, kwargs['n_layers'] + 1):
            layers[f'layer_{layer_idx}'] = get_layers(
                model='rect_kan',
                grid_size=kwargs['grid_size'], grid_min=kwargs['grid_min'], grid_max=kwargs['grid_max'],
                width=kwargs['width'],
                grad_width=kwargs['grad_width'],
                in_channels=kwargs['in_channels'] if layer_idx == 1 else kwargs[f'hidden_channels_{layer_idx - 1}'],
                out_channels=kwargs['out_channels'] if layer_idx == kwargs['n_layers'] else kwargs[f'hidden_channels_{layer_idx}'],
                kernel_size=kwargs[f'kernel_size_{layer_idx}'],
                stride=kwargs[f'stride_{layer_idx}'],
            )['layer_1']

    return layers
