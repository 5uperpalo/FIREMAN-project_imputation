import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def gain_data_prep(data_missing):
    """Prepare input data with missing values as np.nan
    for GAIN generator. Generate a tensor data and
    corresponding mask locating missing values by 0s.

    Args:
        data_missing(np.array): input missing data with NaN values

    Returns:
        data_missing(pytorch.tensor): converted input
        mask(pytorch.tensor): mask of data_missing, 0s mark missing values
    """
    data_missing = data_missing.copy() 
    mask = np.isnan(data_missing)
    data_missing[mask] = 0

    mask = 1-mask
    mask = mask.astype(np.int)

    return torch.from_numpy(data_missing), torch.from_numpy(mask)


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
    p(float): probability, <0,1>
    rows(int): the number of rows
    cols(int): the number of columns

    Returns:
    binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix


def mcar_gen(data, probability):
    '''Generate Missing Completely At Random data with given
    probability from input data.

    Args:
    data(np.array): input data

    Returns:
    data_missing(np.array): data with missing values
    mask(np.array): mask of data_mising, 0s mark missing values
    '''
    no, dim = data.shape
    mask = binary_sampler(1-probability, no, dim)
    data_missing = data.copy()
    data_missing[mask == 0] = np.nan
    return data_missing, mask


def cust_dataloader(data_missing, mask, batch_size, device, shuffle=True):
    '''Helper function to load and shuffle tensors into models in
    batches

    Args:
    data_missing(np.array): data with missing values
    mask(np.array): mask of data_mising, 0s mark missing values
    batch_size(int): size of batch
    device(str): cuda or cpu to be used for computation
    shuffle(boolean): if the tensors should be shuffled before load

    Returns:
    DataLoader: PyTorch DataLoader object
    '''
    # added .float() as I was getting expected scalar type Float but found Double (numpy stores as Double? https://discuss.pytorch.org/t/pytorch-why-is-float-needed-here-for-runtimeerror-expected-scalar-type-float-but-found-double/98741)
    data_missing = torch.Tensor(data_missing).float().to(device)
    mask = torch.Tensor(mask).float().to(device)
    tensor_data_mask = TensorDataset(data_missing, mask)
    return DataLoader(tensor_data_mask, batch_size=batch_size, shuffle=shuffle)


def init_weights(NetModel):
    '''Helper function to initialize model weights. By default
    the weights are initialized automatically by Kaiming He:
    https://stackoverflow.com/a/56773737/8147433
    how to use: NetModel.apply(init_weights)

    Args:
    NetModel(nn.Module): torch module

    Returns:
    NetModel(nn.Module): torch module with initilized weights
    '''
    if type(NetModel) == nn.Linear:
        # maybe weight.data?, also maybe gain?
        # https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156/5
        # as we are using relu....
        torch.nn.init.xavier_normal_(NetModel.weight.data, gain=nn.init.calculate_gain('relu'))
        # in original GAIN post it is 0 but I have also seen 0.01
        NetModel.bias.data.fill_(0)
        # torch.nn.init.xavier_normal_(NetModel.bias.data)
