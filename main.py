from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import os
from model import *
from DataHelper import *
import torch.utils.data as data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
hidden_size = 256
#The number of vocabulary
vocab_size = 30
teacher_forcing_ratio = 0
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05
MAX_LENGTH = 16
batch_size = 8
tense_size = 2

vocab = Vocabulary()
vocab.build_vocab('./lab4_dataset/train.txt')
data_transformer = DataTransformer('./lab4_dataset/train.txt')

tenser = TenseRNN(tense_size= tense_size, device= device).to(device)
encoder = EncoderRNN(input_size= vocab_size, hidden_size= hidden_size, device= device).to(device)
mid = midRNN(input_size= hidden_size, latent_size= latent_size, tense_size= tense_size).to(device)
decoder = DecoderRNN(hidden_size= hidden_size, output_size=vocab_size, device= device).to(device)


def train(tenser, encoder, mid, decoder, tenser_optimizer, encoder_optimizer, mid_optimizer, decoder_optimizer ,step, input_tensor, target_tensor):
    word_length = input_tensor.size(1)

    tenser_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    mid_optimizer.zero_grad()
    decoder_optimizer.zero_grad()    

    # transpose tensor from (batch_size, seq_len) to (seq_len, batch_size)
    
    input_tensor = input_tensor.transpose(0, 1).to(device)
    target_tensor = target_tensor.transpose(0, 1).to(device)
    input_tense = tenser(input_tensor[0]).to(device)
    target_tense = tenser(target_tensor[0]).to(device)

    hidden = encoder.initHidden(batch_size, tense_size)
    hidden = torch.cat((hidden, input_tense),dim=2)
            
    for ei in range(1, word_length):
        encoder_output, hidden = encoder(
        input_tensor[ei], hidden)        
    hidden = mid(hidden, target_tense)
    decoder_input = torch.LongTensor([[SOS_token] for i in range(batch_size)]).to(device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #----------sequence to sequence part for decoder----------#
    loss = 0
    # Teacher forcing: Feed the target as the next input
    for di in range(1, word_length):
        decoder_output, hidden = decoder(
            decoder_input, hidden) 

        if use_teacher_forcing:
            decoder_input = target_tensor[di]  # Teacher forcing
        else:
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

        pred = decoder_output
        
        loss += loss_function(pred.view(batch_size,-1), target_tensor[di], mid.mu, mid.logvar)
    loss.backward()
    tenser_optimizer.step()
    encoder_optimizer.step()
    mid_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / word_length
def evaluate(tenser, encoder, mid, decoder,test_loader):    
    prediction = []
    targets = []
    inputs = []

    with torch.no_grad():    
        for input_tensor, target_tensor in test_loader:
            batch_size = input_tensor.size(0)
            hidden = encoder.initHidden(batch_size, tense_size)
            
            # convert input&target into string
            for idx in range(batch_size):
                targets.append(vocab.indices_to_sequence(target_tensor[idx].cpu().data.numpy()))
                inputs.append(vocab.indices_to_sequence(input_tensor[idx].cpu().data.numpy()))

            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # transpose tensor from (batch_size, seq_len) to (seq_len, batch_size)
            input_tensor = input_tensor.transpose(0, 1)
            target_tensor = target_tensor.transpose(0, 1)
            input_tense = tenser(input_tensor[0])
            target_tense = tenser(target_tensor[0])
            # calculate number of time step
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)
            hidden = torch.cat((hidden, input_tense), dim=2)
            
            #----------sequence to sequence part for encoder----------#
            for ei in range(1, input_length):
                output, hidden = encoder(
                    input_tensor[ei], hidden)
        
            decoder_input = torch.tensor([SOS_token for i in range(batch_size)], device=device)
            
            hidden = mid(hidden, target_tense)
            
            #----------sequence to sequence part for decoder----------#
            for di in range(1, input_length):
                decoder_output, hidden = decoder(
                    decoder_input, hidden) 
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
            

                #get predict indices
                if(di != 1):
                    output = torch.cat((output, decoder_input), dim=0)
                else:
                    output = decoder_input 
            output = output.view(batch_size, -1)  
        # convert indices into string
            for idx in range(batch_size):
                prediction.append(vocab.indices_to_sequence(output[idx].cpu().data.numpy()))
    return prediction
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    criterion = nn.NLLLoss(ignore_index=2, size_average= True, reduction= 'mean')
    recon_loss = criterion(recon_x, x)
    kl_loss = KLD_weight * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    return recon_loss - kl_loss

if __name__ == "__main__":
    start = time.time()
    plot_losses = []
    plot_scores = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    train_set = EncoDataSet(data_transformer, 'train')
    test_set = EncoDataSet(data_transformer, 'test')
    train_loader = data.DataLoader(train_set, batch_size= batch_size, shuffle=True, num_workers=4, drop_last= True)
    test_loader = data.DataLoader(test_set, batch_size= batch_size, num_workers=4, drop_last= True)

    tenser_optimizer = optim.SGD(tenser.parameters(), lr= LR)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
    mid_optimizer = optim.SGD(mid.parameters(), lr= LR)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)
    for epoch in range(10):
        lowest_loss = 2
        loss = 0
        for step, (input_tensor, target_tensor) in enumerate(train_loader):
            input_tensor.to(device)
            target_tensor.to(device)

            loss += train(tenser, encoder, mid, decoder, tenser_optimizer, 
                            encoder_optimizer, mid_optimizer, 
                            decoder_optimizer ,step, input_tensor, target_tensor)
        loss /= step
        if loss < lowest_loss:
            lowest_loss = loss
            torch.save({
            'tenser_state_dict': tenser.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'mid_state_dict': mid.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'tenser_optimizer_state_dict': tenser_optimizer.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'mid_optimizer_state_dict': mid_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict()
            }, "./model/"+str(loss)+'.pt')
        print("the ", epoch," epochs Loss:", loss)
        pred = evaluate(tenser, encoder, mid, decoder, test_loader)
        # print(pred)

        # print(pred)

            