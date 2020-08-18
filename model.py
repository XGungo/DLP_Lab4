
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 30
teacher_forcing_ratio = 1.0
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05
MAX_LENGTH = 10
latent_size = 32
class TenseRNN(nn.Module):
    def __init__(self, tense_size):
        super(TenseRNN, self).__init__()
        self.tense_size = tense_size
        self.embedding = nn.Embedding(4, self.tense_size)
    def forward(self, input):
        output = self.embedding(input).view(1, -1, self.tense_size)
        return output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size)
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def initHidden(self, batch_size, tense_size):
        return torch.zeros(1, batch_size, self.hidden_size - tense_size, device=self.device)

class midRNN(nn.Module):
    def __init__(self, input_size, latent_size, tense_size):
        super(midRNN, self).__init__()
        self.fc1 = nn.Linear(input_size, latent_size)
        self.fc2 = nn.Linear(input_size, latent_size)
        self.fc3 = nn.Linear(latent_size+tense_size, input_size)
        self.relu = nn.ReLU(inplace= True)
        self.context_vector = None
        self.mu = None
        self.logvar = None
    def forward(self, hidden, tense):
        self.mu = self.fc1(hidden)
        self.logvar = self.fc2(hidden)
        self.context_vector = reparameterize(self.mu, self.logvar)
        latent_cat_tense = torch.cat((self.context_vector, tense), dim= 2)
        hidden = self.fc3(latent_cat_tense)
        hidden = self.relu(hidden)

        return hidden
#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output)
        # print(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std
if __name__ == "__main__":
    encoder = EncoderRNN(input_size=vocab_size, hidden_size= hidden_size, latent_size= latent_size)
    decoder = DecoderRNN(hidden_size=hidden_size, output_size= vocab_size)
    data_transformer = DataTransformer('./lab4_dataset/train.txt')
    print(encoder)
    print(decoder)
