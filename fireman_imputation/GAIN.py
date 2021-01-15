import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import  PopulationBasedTraining
import numpy as np
import torch
from .src import utils
from .src import netGAIN
# from torch.nn.parallel import DistributedDataParallel


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
    checkpoint_dir = config['checkpoint_dir']
    use_cuda = config.get("use_gpu") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # fill missing values with 0
    data_missing = data_train.copy()
    data_missing[mask_train] = 0

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
                                                hint_rate, alpha, epochs, show_prog=True,
                                                checkpoint_dir=checkpoint_dir)

    return lossG_curr, lossD_curr, lossMSE_train_curr, netG, netD, rmse


'''Raytune
# config example
raytune_config = {
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([2, 4, 8, 16]),
    'hint_rate':,
    'alpha':,
    'epochs':,
    'checkpoint_dir':'',
    }
'''
def GAINtune_main(raytune_config, data_train, mask_train, data_test, mask_test, tune_iter=5):
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

    ''' NOT TESTED PBT approach
    scheduler = PopulationBasedTraining(
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "netG_lr": lambda: np.random.uniform(1e-2, 1e-5),
            "netD_lr": lambda: np.random.uniform(1e-2, 1e-5),
        })
    '''

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=['rmse', 'lossG', 'lossD', 'lossMSE', 'training_iteration'])

    analysis = tune.run(
        tune.with_parameters(GAIN_main,
                             data_train=data_train, mask_train=mask_train,
                             data_test=data_test, mask_test=mask_test,
                             raytune=True),
        name='GAIN',
        verbose=1,
        stop={
            'training_iteration': tune_iter,
        },
        metric='rmse',
        mode='min',
        scheduler=scheduler,
        progress_reporter=reporter,
        config=raytune_config)

    ''' I need to troubleshoot this part first
    # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/pbt_dcgan_mnist/pbt_dcgan_mnist_func.py
    best_trial = analysis.get_best_trial("rmse", "min", "last")
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial final validation loss: {}'.format(
        best_trial.last_result['rmse']))
    print('Best trial final lossG: {}'.format(
        best_trial.last_result['lossG']))
    

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    '''
    return analysis