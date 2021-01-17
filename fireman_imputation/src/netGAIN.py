from ray import tune
import torch
from torch import nn
from . import utils
import numpy as np
from sklearn import metrics
import os

'''
Define loss functions criterions
https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
- reduction can be mean or sum, mean was in github implementation - original paper used 'sum'
'''
reduction = 'sum'
disc_criterion = nn.BCELoss(reduction=reduction)
gen_criterion = nn.MSELoss(reduction=reduction)


def get_gain_net_block(input_dim, output_dim):
    '''Simple helper function that return [Linear, ReLU] 
    block for Net module.

    Args:
        input_dim: input block dimension
        output_dim: out block dimension

    Returns:
        block(nn.Sequential): converted input
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    '''Create GAIN generator Net with 3 layers. Input, hidden, output.

    Args:
        input_dim: number of neurons in input layer(size of input data)
        hidden_dim: number of neurons in hidden layer

    Returns:
        Net(nn.Module): generator Net
    '''
    def __init__(self, input_dim, hidden_dim):
        super(Generator, self).__init__()
        # Build the neural network
        self.net = nn.Sequential(
            get_gain_net_block(input_dim*2, hidden_dim),
            get_gain_net_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, data_w_noise, mask):
        input_data = torch.cat(tensors=[data_w_noise, mask], dim=1).float()
        return self.net(input_data)


class Discriminator(nn.Module):
    '''Creates GAIN discriminator Net with 3 layers. Input, hidden, output.

    Args:
        input_dim: number of neurons in input layer(size of input data)
        hidden_dim: number of neurons in hidden layer

    Returns:
        Net(nn.Module): generator Net
    '''
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # Build the neural network
        self.net = nn.Sequential(
            get_gain_net_block(input_dim*2, hidden_dim),
            get_gain_net_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, imputed_data, hint_matrix):
        input_data = torch.cat(tensors=[imputed_data, hint_matrix], dim=1).float()
        return self.net(input_data)


def lossG(netG, netD, data, mask, data_w_noise, hint_matrix, alpha, reduction=reduction, gen_criterion=gen_criterion):
    '''Generator loss function
    '''
    # Generator
    generator_output = netG(data_w_noise, mask)
    # Combine with original data
    imputed_data = data_w_noise * mask + generator_output * (1-mask)
    # Discriminator
    D_prob = netD(imputed_data, hint_matrix)
    # Loss
    if reduction == 'mean':
        G_loss1 = -torch.mean((1-mask) * torch.log(D_prob + 1e-8))
    elif reduction == 'sum':
        G_loss1 = -torch.sum((1-mask) * torch.log(D_prob + 1e-8))
    else:
        print('Not implemented')

    MSE_train_loss = gen_criterion(mask * generator_output, mask * data_w_noise)
    G_loss = G_loss1 + alpha * MSE_train_loss
    return G_loss, MSE_train_loss


def lossD(netG, netD, mask, data_w_noise, hint_matrix, disc_criterion=disc_criterion):
    '''Discriminator loss function
    '''
    # Since the generator is needed when calculating the discriminator's loss, we have to
    # call .detach() on the generator result to ensure that only the discriminator is updated!
    # related to: https://stackoverflow.com/a/58699937/8147433
    generator_output = netG(data_w_noise, mask).detach()
    # Combine with original data
    imputed_data = data_w_noise * mask + generator_output * (1-mask)

    D_prob = netD(imputed_data, hint_matrix)
    D_loss = disc_criterion(D_prob, mask)

    return D_loss


def trainGAIN(netG, netD, optimG, optimD,
              dataloader, data_test, mask_test,
              hint_rate, alpha, epochs, show_prog=False, raytune=False, checkpoint_dir=None):
    '''Procedure to train GAIN network

    Args:
        netG: Generator Net 
        netD: Discriminator Net
        optimG: Generator optimization function
        optimD: Discriminator optimization function
        dataloader: PyTorch DataLoader object
        data_test(numpy array): test data
        mask_test(numpy array): test data mask(1 = imputed value)
        hint_rate: per orig. GAIN paper 
        alpha: per orig. GAIN paper
        epochs: number of training epochs
        show_prog: show intermidiate progress True/False

    Returns:
        (tuple):
            - lossG: last Generator loss value 
            - lossD: last Discriminator loss value
            - lossMSE: last Generator MSE value
            - netG(nn.Module): trained Generator
            - netD(nn.Module): trained Discriminator
            - rmse: RMSE of test data
    '''
    if checkpoint_dir != '':
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        netG.load_state_dict(checkpoint["netGmodel"])
        netD.load_state_dict(checkpoint["netDmodel"])
        optimG.load_state_dict(checkpoint["optimG"])
        optimD.load_state_dict(checkpoint["optimD"])

    for epoch in range(epochs):
        for i, (data, mask) in enumerate(dataloader):
            # /100 as noise was added in the original paper from uniform distribution <0,0.01>
            noise = (1-mask) * torch.rand(mask.shape)/100
            hint_matrix = utils.HINTmatrix_gen(mask, hint_rate, orig_paper=False)
            data_w_noise = data + noise

            optimD.zero_grad()
            lossD_curr = lossD(netG, netD, mask, data_w_noise, hint_matrix)
            if show_prog is True:
                print('lossD_curr: ' + str(lossD_curr))
            lossD_curr.backward(retain_graph=True)
            optimD.step()

            optimG.zero_grad()
            lossG_curr, lossMSE_train_curr = lossG(netG, netG, data, mask, data_w_noise, hint_matrix, alpha)
            lossG_curr.backward(retain_graph=True)
            optimG.step()
            
            if show_prog is True:
                print('lossG_curr: ' + str(lossG_curr.item()))
                print('lossMSE_train_curr: ' + str(lossMSE_train_curr.item()))
                if i % 100 == 0:
                    print('Iter: {}'.format(i))
                    print('Train_loss: {:.4}'.format(np.sqrt(lossMSE_train_curr.item())))
                    print()
        
        rmse = GAIN_rmse(netG, data_test, mask_test) 
        
        if raytune is True:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save({
                    'netDmodel': netD.state_dict(),
                    'netGmodel': netG.state_dict(),
                    'optimD': optimD.state_dict(),
                    'optimG': optimG.state_dict(),
                    'epoch': epoch,
                }, path)
            tune.report(lossG=lossG_curr.item(), lossD=lossD_curr.item(), 
                        lossMSE=lossMSE_train_curr.item(), rmse=rmse)

    return lossG_curr.item(), lossD_curr.item(), lossMSE_train_curr.item(), netG, netD, rmse


def GAIN_rmse(netG, data, mask):
    '''Procedure to test GAIN generator network performance using RMSE

    Args:
        netG: Generator Net 
        data: input data(numpy array)
        mask: mask marking data that will be imputed(numpy array)

    Returns:
        rmse: Root Means Square Error between original and imputed data
    '''
    # transform test data to tensor and forward it through generator
    data_missing_torch, mask_torch = utils.dataPrepGAIN(data, mask)
    data_imputed = netG(data_missing_torch, mask_torch)
    data_imputed = data_imputed.detach().numpy()

    # merge the imputed data(zero out rest in imputed data) and data with missing values
    # data_missing_0 = data.copy()
    # data_missing_0[mask==0] = 0
    # data_imputed = (1-mask)*data_imputed + data_missing_0
    data_imputed = (1-mask)*data_imputed + mask*data

    # compute error
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    rmse = metrics.mean_squared_error(data, data_imputed, squared=False)
    return rmse
