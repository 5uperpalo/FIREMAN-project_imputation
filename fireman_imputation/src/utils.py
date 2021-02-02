import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import random


def dataPrepGAIN(data, mask):
    '''Prepare input data with missing values as np.nan
    for GAIN generator. Generate a tensor data and
    corresponding mask locating missing values by 0s.

    Args:
        data(np.array): input data
        mask(np.array): mask with values that are missing

    Returns:
        (tuple):
            - data_missing(pytorch.tensor): converted input
            - mask(pytorch.tensor): mask of data_missing, 0s mark missing values
    '''
    data_missing = data.copy() 
    data_missing[mask==0] = 0

    return torch.from_numpy(data_missing), torch.from_numpy(mask)


def HINTmatrix_gen(mask, hint_rate, orig_paper=True):
    '''Generate Hint matrix for Discriminator with the given
    hint_rate that defines fraction of mask that is copied to hint_matrix.
    More aboout hinting mechanism implementation vs orig. paper:
    https://github.com/jsyoon0823/GAIN/issues/2

    Args:
        mask(np.array):
        hint_rate(float): probability, <0,1>

    Returns:
        hint_matrix(np.array)
    '''
    no, dim = mask.shape
    unif_random_matrix = np.random.uniform(0., 1., size=[no, dim])

    if orig_paper is True:
        hint_matrix = (unif_random_matrix >= hint_rate).astype(int) * 0.5
        hint_matrix = (unif_random_matrix < hint_rate).astype(int) * mask.numpy() + hint_matrix
    else:
        hint_matrix = np.zeros([no, dim])
        hint_matrix = (unif_random_matrix < hint_rate).astype(int) * mask.numpy() + hint_matrix
    return torch.from_numpy(hint_matrix)


def MCARgen(data, probability):
    '''Generate Missing Completely At Random(MCAR) data with given
    probability.

    Args:
        data(np.array): input data
        probability(float): probability, <0,1>

    Returns:
        (tuple):
            - data_missing(np.array): data with missing values
            - mask(np.array): mask of data_mising, 0s mark missing values
    '''
    no, dim = data.shape
    
    unif_random_matrix = np.random.uniform(0., 1., size=[no, dim])
    mask = (unif_random_matrix > probability).astype(int)

    data_missing = data.copy()
    data_missing[mask == 0] = np.nan

    return data_missing, mask


def MCARgen_cont(data, probability, cont_segments, cont_segments_distrib=None):
    '''Generate MCAR data with continuous segments with predefined probabilities.

    Args:
        data(np.array): input data
        probability(float): probability, <0,1>
        cont_segments(list): list of continuous segment sizes 
        cont_segments_distrib(list): probabilities of segments sizes, if not set -> uniform

    Returns:
        (tuple):
            - data_missing(np.array): data with missing values
            - mask(np.array): mask of data_mising, 0s mark missing values
    '''
    data_missing, mask = MCARgen(data, probability)
    no, dim = mask.shape
    # replace 1s with 0s and vice versa so we can easily track number of missing values by sum()
    mask = 1-mask
    ones_per_col = sum(mask)
    temp = 0
    new_mask = []
    for sum_col_ones in ones_per_col:
        # keeps temporary count of ones - there can be leftover ones from previous column
        temp += sum_col_ones
        # create "large enough" numpy array with given disctribution of given continuous segment sizes
        cont_segments_values = np.array(random.choices(cont_segments, cont_segments_distrib, k=no))
        # filter out segment sizes which cummulative sum is max number of 1s in a column
        cont_segments_values = cont_segments_values[cont_segments_values.cumsum()<=temp]
        # create temporary column mask
        temp_new_mask = np.zeros(no)
        for val in cont_segments_values:
            if temp-val>=0:
                # find the indices with 0s, shuffle them and iterate over them
                zero_indices = np.where(temp_new_mask==0)[0]
                random.shuffle(zero_indices)
                for zero_ind in zero_indices:
                    # if there is enough 0s to make continuous 1s segment + at least 1 zero
                    # before and after segment(or they are at the beginning/end) so we have
                    # non-connecting/non-overlapping segments
                    # + the size of segment must be the size of "val"
                    if (sum(temp_new_mask[zero_ind:(zero_ind+val+1)])==0) and (sum(temp_new_mask[(zero_ind-1):(zero_ind+val)])==0) and (len(temp_new_mask[zero_ind:(zero_ind+val)])==val):
                        temp_new_mask[zero_ind:(zero_ind+val)] = 1
                        temp-=val
                        break
        new_mask.append(temp_new_mask)
    
    new_mask = np.array(new_mask).T
    # replace 1s with 0s back so 0s mark missing values
    new_mask = 1 - new_mask
    
    data_missing = data.copy()
    data_missing[new_mask == 0] = np.nan

    return data_missing, new_mask


def dataloaderCust(data_missing, mask, batch_size, device, shuffle=True):
    '''Helper function to load and shuffle tensors into models in
    batches.

    Args:
        data_missing(np.array): data with missing values
        mask(np.array): mask of data_mising, 1s mark missing values
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

    Args:
        NetModel(nn.Module): torch module

    Returns:
        NetModel(nn.Module): torch module with initilized weights

    Example:
        NetModel.apply(init_weights)
    '''
    if type(NetModel) == nn.Linear:
        # maybe weight.data?, also maybe gain?
        # https://discuss.pytorch.org/t/how-to-fix-define-the-initialization-weights-seed/20156/5
        # as we are using relu....
        torch.nn.init.xavier_normal_(NetModel.weight.data, gain=nn.init.calculate_gain('relu'))
        # in original GAIN post it is 0 but I have also seen 0.01
        NetModel.bias.data.fill_(0)
        # torch.nn.init.xavier_normal_(NetModel.bias.data)
