import os
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import torch
from .src import utils
from .src import netGAIN
# from torch.nn.parallel import DistributedDataParallel


def GAIN_netG_forward(netG, data, missing_val=None):
    '''Procedure to test GAIN generator network perfoarmance 

    Args:
        netG: Generator Net 
        data(numpy array): input data with missing values
        missing_str: string that marks missing values([0,Nan,...])

    Returns:
        data_imputed(numpy array): imputed input data
    '''
    # transform test data to tensor and forward it through generator
    if missing_val == None:
        mask = 1- np.isnan(data).astype(int)
    else:
        mask = (data!=missing_val).astype(int)
    data_missing_torch, mask_torch = utils.dataPrepGAIN(data, mask)
    data_imputed = netG(data_missing_torch, mask_torch).detach().numpy()

    # merge the imputed data(zero out rest in imputed data) and data with missing values
    data_imputed = (1-mask)*data_imputed + mask*data

    return data_imputed


def GAIN_main(config, data_train, mask_train, data_test, mask_test, show_prog=False, raytune=False, cont=False, **kwargs):
    '''Full training procedure.

    Args:
        config(dict): dictionary with config parameters
        data_train: input data(numpy array)
        mask_train: binary mask marking points that will be imputed (numpy array)
        data_test: input data(numpy array)
        mask_test: binary mask marking points that will be imputed (numpy array)
        show_prog(boolean): if the progress should be printed
        raytune(boolean): for the hyperparameter tuning using Raytune
        cont(boolean): if the preexisting models should go through extended training,
                       in that case specify model parameters in **kwargs

    Returns:
        (tuple):
            - netG(nn.Module): Generator network
            - netD(nn.Module): Discriminator network
    '''
    batch_size = config['batch_size']
    hint_rate = config['hint_rate']
    alpha = config['alpha']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    checkpoint_dir = ''
    use_cuda = config.get('use_gpu') and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # fill missing values with 0
    data_missing = data_train.copy()
    data_missing[mask_train==0] = 0

    input_dim = data_missing.shape[1]

    if cont is False:
        # initialize your generator, discriminator, and optimizers
        netG = netGAIN.Generator(input_dim, input_dim)
        netD = netGAIN.Discriminator(input_dim, input_dim)
        netG.apply(utils.init_weights)
        netD.apply(utils.init_weights)
    else:
        # load generator and discriminator from the dictionary in **kwargs
        netG = kwargs.get('netG')
        netD = kwargs.get('netD')

    netG.to(device)
    netD.to(device)
    # not tested https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html # noqa
    # if device=='cuda' and torch.cuda.device_count() > 1:
    #     netG = DistributedDataParallel(netG)
    #     netD = DistributedDataParallel(netD)

    # Note: each optimizer only takes the parameters of one particular model,
    # since we want each optimizer to optimize only one of the model
    optimG = torch.optim.Adam(netG.parameters(), lr=learning_rate)
    optimD = torch.optim.Adam(netD.parameters(), lr=learning_rate)

    # create tensor dataset that includes data and masks
    dataloaderGAIN = utils.dataloaderCust(data_missing, mask_train, batch_size, device)

    lossG_curr, lossD_curr, lossMSE_train_curr, netG, netD, rmse = netGAIN.trainGAIN(
                                                netG, netD, optimG, optimD,
                                                dataloaderGAIN, data_test, mask_test,
                                                hint_rate, alpha, epochs, show_prog=show_prog,
                                                raytune=raytune, checkpoint_dir=checkpoint_dir)

    return lossG_curr, lossD_curr, lossMSE_train_curr, netG, netD, rmse


def GAINtune_main(raytune_config, data_train, mask_train, data_test, mask_test):
    '''GAIN Tunning with Raytune.

    Args:
        raytune_config(dict): dictionary with config parameterr arrays
        data_train: input data(numpy array)
        mask_train: binary mask marking points that will be imputed (numpy array)
        data_test: input data(numpy array)
        mask_test: binary mask marking points that will be imputed (numpy array)
        tune_iter: stopping condition
    
    Returns:
        netG(nn.Module): Best Generator network
    '''

    scheduler = ASHAScheduler(
        metric="rmse",
        mode="min",
        max_t=5000,
        grace_period=1,
        reduction_factor=2)

    analysis = tune.run(
        tune.with_parameters(GAIN_main,data_train=data_train, mask_train=mask_train,
                                data_test=data_test, mask_test=mask_test, raytune=True),
        name='GAIN',
        verbose=1,
        scheduler=scheduler,
        config=raytune_config)

    best_trial = analysis.get_best_trial("rmse", "min", "last")
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial RMSE: {}'.format(
        best_trial.last_result['rmse']))
    print('Best trial lossG: {}'.format(
        best_trial.last_result['lossG']))    

    input_dim = data_train.shape[1]
    best_trained_netG = netGAIN.Generator(input_dim, input_dim)

    best_checkpoint_dir = best_trial.checkpoint.value
    best_model_checkpoint = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_netG.load_state_dict(best_model_checkpoint['netGmodel'])
    
    return best_trained_netG
