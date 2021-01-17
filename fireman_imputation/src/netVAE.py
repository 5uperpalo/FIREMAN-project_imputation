class AutoencoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # Build the neural network
        self.net = nn.Sequential(
            get_gain_net_block(input_dim*2, hidden_dim),
            get_gain_net_block(hidden_dim, hidden_dim),
            get_gain_net_block(hidden_dim, hidden_dim),
            get_gain_net_block(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.LeakyReLU(),
        )

    def forward(self, imputed_data, hint_matrix):
        '''
        Function for completing a forward pass of the AutoencoderNet
        '''
        input_data = torch.cat(tensors=[imputed_data, hint_matrix], dim=1).float()
        return self.net(input_data)