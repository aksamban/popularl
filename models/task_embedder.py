import torch.nn as nn
import torch

torch.manual_seed(42)

class TaskEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TaskEmbedder, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x has shape : (batch_size, seq_len, input_dim)
        # batch_size : 1 --> only one trajectory
        # seq_len : episode length of previous epsiode
        # input_dim : 1 or 2 (o_t) or (o_t, a_t)
        _, hidden = self.rnn(x)  # hidden shape : (1, batch_size, hidden_dim)
        embedding = self.fc(hidden.squeeze(0))  # (batch_size, output_dim)
        return embedding

    def embed_dim(self):
        return self.output_dim