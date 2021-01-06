import numpy as np
import torch
from .src import utils
from .src import gain_net
# from torch.nn.parallel import DistributedDataParallel


def gain_train(gain_params, data_missing, show_prog=False, cont=False, **kwargs):
    batch_size = gain_params['batch_size']
    hint_rate = gain_params['hint_rate']
    alpha = gain_params['alpha']
    epochs = gain_params['epochs']
    learning_rate = gain_params['learning_rate']

    # create a mask based on np.nan values in the data
    # according to paper 0 marks place where data is missing
    mask = np.invert(np.isnan(data_missing))
    mask = mask.astype(np.int)

    # fill missing values with 0
    data_missing[np.isnan(data_missing)] = 0

    input_dim = data_missing.shape[1]

    if cont is False:
        # initialize your generator, discriminator, and optimizers
        gen = gain_net.Generator(input_dim, input_dim)
        disc = gain_net.Discriminator(input_dim, input_dim)
    else:
        # load generator and discriminator from the dictionary in **kwargs
        gen = kwargs.get('gen')
        disc = kwargs.get('disc')

    if torch.cuda.is_available():
        device = 'cuda:0'
        gen.to(device)
        disc.to(device)
        # not tested https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html # noqa
        # if torch.cuda.device_count() > 1:
        #     gen = DistributedDataParallel(gen)
        #     disc = DistributedDataParallel(disc)
    else:
        device = 'cpu'
        gen.to(device)
        disc.to(device)

    gen.apply(utils.init_weights)
    disc.apply(utils.init_weights)
    # Note: each optimizer only takes the parameters of one particular model,
    # since we want each optimizer to optimize only one of the model
    gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate)

    # create tensor dataset that includes data and masks
    data_torch = utils.cust_dataloader(data_missing, mask, batch_size, device)

    for epoch in range(epochs):
        for i, (data, mask) in enumerate(data_torch):
            # /100 as noise was added in the original paper from uniform distribution <0,0.01>
            noise = (1-mask) * torch.rand(mask.shape)/100
            hint_matrix = mask * utils.binary_sampler(hint_rate, mask.shape[0], mask.shape[1])
            data_w_noise = data + noise

            disc_opt.zero_grad()
            D_loss_curr = gain_net.discriminator_loss(gen, disc, mask, data_w_noise, hint_matrix)
            if show_prog is True:
                print('D_loss_curr: ' + str(D_loss_curr))
            D_loss_curr.backward(retain_graph=True)
            disc_opt.step()

            gen_opt.zero_grad()
            G_loss_curr, MSE_train_loss_curr = gain_net.generator_loss(gen, disc, data, mask, data_w_noise, hint_matrix, alpha)
            if show_prog is True:
                print('G_loss_curr: ' + str(G_loss_curr))
                print('MSE_train_loss_curr: ' + str(MSE_train_loss_curr))
            G_loss_curr.backward(retain_graph=True)
            gen_opt.step()

            if show_prog is True:
                if i % 100 == 0:
                    print('Iter: {}'.format(i))
                    print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
                    print()
    return gen, disc
