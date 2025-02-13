import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, rnn_type='GRU'):
        super(Encoder, self).__init__()

        # Define RNN Type
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # To take Hidden State to mu and logvar
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Run RNN
        rnn_out, hidden_out = self.rnn(x)

        # Find mu and logvar
        mu = self.mu(hidden_out.squeeze(0))
        var = self.var(hidden_out.squeeze(0))

        return mu, var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, rnn_type='GRU'):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim)

        # Define RNN Type
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.recon = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, seq_len):
        z = self.fc(z).unsqueeze(1)
        z = z.repeat(1, seq_len, 1)
        rnn_out, _ = self.rnn(z)
        recon_x = self.recon(rnn_out)
        return recon_x


class VAE(nn.Module):
    def __init__(self, input_dim, h1_dim, l1_dim, embedding_dim, h2_dim, output_dim):
        super(VAE, self).__init__()
        l2_dim = l1_dim+embedding_dim
        self.encoder = Encoder(input_dim, h1_dim, l1_dim)
        self.decoder = Decoder(l2_dim, h2_dim, output_dim)
        # output_dim should be 1+input_dim as o_t, r_t and a_t

    def encode(self, x):
        mu, var = self.encoder(x)
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z, mu, var

    def decode(self, z, seq_len):
        return self.decoder(z, seq_len)

    def forward(self, x, c, seq_len):
        # here x is o_t, a_t
        # c has shape (1, embedding_size)
        
        # get latent space
        z, mu, var = self.encode(x)

        # concat z and c
        # c should have have shape (batch_size, embedding_size)
        # c = c.repeat(z.shape[0], 1)
        z_c = torch.cat((z.unsqueeze(0), c), dim=1)
        # reconstruct
        recon_x = self.decode(z_c, seq_len)

        return recon_x, mu, var

    def latent(self, x):
        z, mu, var = self.encode(x)
        return z

    def loss(self, recon_x, x, mu, var, beta):  # here x is o_t, a_t and r_t
        print(recon_x.shape, x.shape)
        
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())

        return recon_loss + (beta * kl_loss)