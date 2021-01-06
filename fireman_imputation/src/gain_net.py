import torch
from torch import nn
# can be mean or sum, mean was in github implementation, sum was in paper
reduction = 'mean'

# per https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
disc_criterion = nn.BCELoss(reduction=reduction)
gen_criterion = nn.MSELoss(reduction=reduction)


def get_gain_net_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
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
        '''
        Function for completing a forward pass of the GainNet
        '''
        input_data = torch.cat(tensors=[data_w_noise, mask], dim=1).float()
        return self.net(input_data)


class Discriminator(nn.Module):
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
        '''
        Function for completing a forward pass of the GainNet
        '''
        input_data = torch.cat(tensors=[imputed_data, hint_matrix], dim=1).float()
        return self.net(input_data)


def discriminator_loss(gen, disc, mask, data_w_noise, hint_matrix, disc_criterion=disc_criterion):
    # Generator
    # from Coursera GAN lectures/notebooks:
    # Since the generator is needed when calculating the discriminator's loss, you will need to
    # call .detach() on the generator result to ensure that only the discriminator is updated!
    # related to: https://stackoverflow.com/a/58699937/8147433
    generator_output = gen(data_w_noise, mask).detach()
    # Combine with original data
    imputed_data = data_w_noise * mask + generator_output * (1-mask)

    D_prob = disc(imputed_data, hint_matrix)
    D_loss = disc_criterion(D_prob, mask)

    return D_loss


def generator_loss(gen, disc, data, mask, data_w_noise, hint_matrix, alpha, reduction=reduction, gen_criterion=gen_criterion):
    # Generator
    generator_output = gen(data_w_noise, mask)
    # Combine with original data
    imputed_data = data_w_noise * mask + generator_output * (1-mask)
    # Discriminator
    D_prob = disc(imputed_data, hint_matrix)
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
